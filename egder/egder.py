# Based on the LiveEdit framework (Chen et al., CVPR 2025).
# EGDER adds evidence-guided localization, dual experts, and quality-aware joint routing.
# See README.md for attribution and LICENSE.upstream for the upstream notice.
from editor.vllms_for_edit.base import BaseVLLMForEdit
from editor.vllm_editors.base import VLLMBaseEditorWithTraining
from utils import find_module, move_to_device
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Tuple
from dataset.vllm import BaseVLLMEditData
from .modules import (mask_regularization_loss, quality_supervision_loss, refine_routing_weights, QVExtractor, LowRankGenerator,
                      EditMaskLocalizer, DualLowRankGenerator,
                      MagnitudeHead, InfluenceHead, StabilityHead,
                      EquivalentResidualComposer, JointRoutingHead)
try:
    from utils.expert_interference_diagnostics import LiveEditDiagnosticsMixin
except ModuleNotFoundError:
    class LiveEditDiagnosticsMixin:
        """Fallback keeps default training/evaluation usable when diagnostics are not installed."""

        def _diagnostics_unavailable(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "Optional diagnostics require utils/expert_interference_diagnostics.py. "
                "Sync that file before calling expert-interference diagnostics.")

        save_expert_diagnostic_snapshot = _diagnostics_unavailable
        load_expert_diagnostic_snapshot = staticmethod(_diagnostics_unavailable)
        collect_routing_diagnostics = _diagnostics_unavailable
        compute_loss_with_manual_experts = _diagnostics_unavailable
        compute_cross_expert_damage = _diagnostics_unavailable
from dataclasses import dataclass
from editor.base import BaseConfig
from torch.optim import Adam
import torch, os, yaml
from torch import nn
import numpy as np

@dataclass
class EGDERConfig(BaseConfig):
    @dataclass
    class TrainConfig:
        lr:float
        lr_cut_it:List[int]
        lr_cut_rate:float
        rel_lambda: float
        gen_lambda: float
        loc_lambda: float
        soft_routing_lambda: float
        hard_routing_lambda: float
        joint_routing_lambda: float = 0.1   # contrastive+null loss on JointRoutingHead (aligned with inference)
        mask_lambda: float = 0.1
        necessity_lambda: float = 0.1
        influence_lambda: float = 0.1
        stability_lambda: float = 0.1
        suppress_lambda: float = 0.05
        magnitude_lambda: float = 0.01
    @dataclass
    class RetrievalEditor:
        module_dim: int
        cross_att_head_n: int
        lora_rank: int
        lora_scale: float
        eqe_n: int # editing query representation
        shortlist_k: int = 0
        mask_ratio: float = 0.5
        mask_temperature: float = 1.0
        necessity_topk: int = 0
        influence_threshold: float = 0.3
        stability_threshold: float = 0.3
        support_weight: float = 1.0
        suppress_weight: float = 1.0
        use_equivalent_residual_expert: bool = True
        # Joint sparse routing config (v2 - multiplicative refinement on top of legacy gating)
        use_joint_routing: bool = True              # True: joint sparse routing; False: legacy hard+soft routing
        preselect_k: int = 32                       # K0: cheap preselection candidate count (on gated experts)
        routing_keep_hard_gate: bool = True         # CRITICAL: keep vis_sim>vis_sim_prot to preserve locality
        routing_keep_abs_sim: bool = True           # CRITICAL: keep sigmoid(sim) gate to preserve relevance suppression
        routing_lambda_vis: float = 1.0             # weight for visual similarity (in preselection)
        routing_lambda_qry: float = 1.0             # weight for query similarity (in preselection only)
        routing_lambda_qua: float = 0.1             # SMALL multiplicative boost from quality (not additive)
        routing_lambda_mask: float = 0.0            # default OFF: infer-time ans_reps=None makes mask unreliable
        routing_tau: float = 1.0                    # temperature for optional sparse normalization
        routing_min_weight: float = 0.0             # 0 = no thresholding (preserve soft behavior)
        routing_use_entmax: bool = False            # reserved; use softmax+threshold by default
        routing_quality_log: bool = False           # use linear quality (more stable than log)
        routing_mask_use_coarse: bool = False       # only used if routing_lambda_mask > 0
        routing_mask_use_necessity: bool = False    # only used if routing_lambda_mask > 0
        # Learnable joint routing head (multiplicative refinement on top of legacy rela*abs)
        use_joint_head: bool = True                 # enable JointRoutingHead refinement at inference
        joint_head_hidden: int = 16                 # hidden dim of the head's small MLP
        joint_head_beta: float = 0.3                # refinement magnitude: multiplier = 1 + beta*tanh(logit) ∈ [1-β, 1+β]
    edit_model_name: str
    retrieval_editor: RetrievalEditor
    train_cfg: TrainConfig
    llm_mid_dim: int
    llm_layer_tmp: str
    edit_layer_i: int

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['train_cfg'] = self.TrainConfig(**data['train_cfg'])
        data['retrieval_editor'] = self.RetrievalEditor(**data['retrieval_editor'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise

class EGDER(VLLMBaseEditorWithTraining, LiveEditDiagnosticsMixin):
    repaired_version = 'edger_repaired_v1'
    '''Lifelong Vision Language Model Editor'''

    def __init__(self, vllm: BaseVLLMForEdit, config:EGDERConfig, device='cuda:0',
            vllm_data_proc: BaseVLLMForEdit = None, data_proc_device = None, verbose = False):
        super().__init__(vllm, config, device)
        self.cfg = config
        self.verbose = verbose
        if vllm_data_proc != None:
            self.vllm_data_proc = vllm_data_proc
            self.vllm_data_proc.model.eval()
            self.vllm_data_proc.model.requires_grad_(False)
            self.data_proc_device = data_proc_device
            self.vllm_data_proc.set_device(data_proc_device)
        # initialize training parameters
        self.sim_scale = 1 / self.cfg.retrieval_editor.module_dim**0.5
        self.vision_tok_n = self.vllm.get_img_token_n()
        (eqe_n, llm_mid_dim, module_dim, cross_att_head_n, lora_rank, lora_scale) = (self.cfg.retrieval_editor.eqe_n, 
            self.cfg.llm_mid_dim, self.cfg.retrieval_editor.module_dim, self.cfg.retrieval_editor.cross_att_head_n, 
            self.cfg.retrieval_editor.lora_rank, self.cfg.retrieval_editor.lora_scale)
        # --- Original modules (kept for backward compatibility / fallback) ---
        self.edit_extractor = QVExtractor(eqe_n, llm_mid_dim, module_dim, cross_att_head_n, self.vision_tok_n, False).to(self.device)
        self.inpt_extractor = QVExtractor(eqe_n, llm_mid_dim, module_dim, cross_att_head_n, self.vision_tok_n, True).to(self.device)
        self.moegen_c = LowRankGenerator(llm_mid_dim, lora_rank, lora_scale, llm_mid_dim, module_dim, cross_att_head_n).to(self.device)
        self.moegen_r = LowRankGenerator(llm_mid_dim, lora_rank, lora_scale, llm_mid_dim, module_dim, cross_att_head_n).to(self.device)
        self.instant_reps_norm = nn.LayerNorm(self.cfg.llm_mid_dim).to(self.device)
        # --- New modules: localized editing pipeline ---
        self.mask_localizer = EditMaskLocalizer(
            llm_mid_dim, module_dim, cross_att_head_n,
            shortlist_ratio=self.cfg.retrieval_editor.mask_ratio).to(self.device)
        self.dual_moegen = DualLowRankGenerator(
            llm_mid_dim, lora_rank, lora_scale,
            llm_mid_dim, module_dim, cross_att_head_n).to(self.device)
        self.magnitude_head = MagnitudeHead(llm_mid_dim, module_dim).to(self.device)
        self.influence_head = InfluenceHead(llm_mid_dim, module_dim).to(self.device)
        self.stability_head = StabilityHead(llm_mid_dim, module_dim).to(self.device)
        self.equiv_composer = EquivalentResidualComposer(lora_rank, module_dim).to(self.device)
        # Learnable joint routing head (identity-initialized; multiplier = 1 at init)
        self.joint_refine_head = JointRoutingHead(
            feat_dim=3,
            hidden_dim=self.cfg.retrieval_editor.joint_head_hidden,
        ).to(self.device)
        # for inference & other hyper-parameters
        self.edit_layer_path = self.cfg.llm_layer_tmp.format(self.cfg.edit_layer_i)
        self.wrap_and_hook()
        self.restore_to_original_model()
        self.set_train(False)
        self.is_editing = False

    def wrap_and_hook(self):
        '''Only for inference stage.'''
        self.now_infer_vt_range = None # [vt_begin, vt_end], The vision token range of current VLLM inference.
        self.now_infer_inpt_embd_shape = None # The shape of current input embedding 
        self.now_infer_query_pos_end = None # The range of query of current input embedding 
        self.manual_edit_residual = None # Optional diagnostics-only residual override.
        # modify VLLM function `get_llm_outpt` for retrieval editor
        def get_llm_outpt_wrap(get_llm_outpt):
            def wrapped_get_llm_outpt(llm_inpt, vt_range):
                if not self.is_train and not self.is_editing and len(self.eqr_pool) > 0:
                    self.now_infer_vt_range = vt_range
                    self.now_infer_inpt_embd_shape = llm_inpt['inputs_embeds'].shape
                    if 'query_range' in llm_inpt:
                        self.now_infer_query_pos_end = llm_inpt['query_range'][1]
                    elif self.verbose:
                        print("Without query range, will use all reps except for visual reps as query reps.")
                return get_llm_outpt(llm_inpt, vt_range)
            return wrapped_get_llm_outpt 
        if not hasattr(self, 'original_get_llm_outpt'):
            self.original_get_llm_outpt = self.vllm.get_llm_outpt
        self.vllm.get_llm_outpt = get_llm_outpt_wrap(self.original_get_llm_outpt)
        # initialize for retrieval editor hooked into the VLLM
        def apply_edit_residual(outpt, edit_residual):
            if isinstance(outpt, (tuple, list)):
                outpt = list(outpt) 
                outpt[0] = outpt[0] + edit_residual
            else:
                outpt = outpt + edit_residual
            return outpt
        def edit_with_moes(module, args, output):
            if self.manual_edit_residual is not None:
                output = apply_edit_residual(output, self.manual_edit_residual)
                self.manual_edit_residual = None
                return output
            if self.is_train:
                output = apply_edit_residual(output, self.train_edit_residual)
                self.train_edit_residual = None
            elif (not self.is_editing and len(self.eqr_pool) > 0 and self.now_infer_vt_range != None):
                reps = output[0] if isinstance(output, (tuple, list)) else output
                assert reps.shape == self.now_infer_inpt_embd_shape 
                vision_reps = reps[:, self.now_infer_vt_range[0]:self.now_infer_vt_range[1]]
                query_reps = reps[:, self.now_infer_vt_range[1]:self.now_infer_query_pos_end] 
                self.now_infer_inpt_embd_shape = self.now_infer_vt_range = self.now_infer_query_pos_end = None
                moe_cs, moe_rs, fuse_coe = self.retrieve_moes(vision_reps, query_reps)
                if moe_cs.shape[0] > 0:
                    edit_residual = self.get_edit_residual(reps, moe_cs, moe_rs, fuse_coe)
                    output = apply_edit_residual(output, edit_residual)
            return output
        edit_layer = find_module(self.vllm.model, self.edit_layer_path)
        if hasattr(edit_layer, 'edit_layer_hooked'): 
            edit_layer._forward_hooks.clear()
        edit_layer.edit_layer_hooked = True
        edit_layer.register_forward_hook(edit_with_moes)

    def retrieve_moes(self, vision_reps:torch.Tensor, query_reps:torch.Tensor, 
                      return_retr_details = False):
        '''Only for inference stage.
        
        When use_joint_routing=True: Uses joint sparse routing with preselection.
        When use_joint_routing=False: Falls back to legacy hard+soft routing.
        '''
        assert len(vision_reps) == len(query_reps) == 1
        
        cfg = self.cfg.retrieval_editor
        m_total = len(self.eqr_pool)
        
        # Empty pool: return empty tensors
        if m_total == 0:
            empty_c = self.moe_cs_pool[:0]
            empty_r = self.moe_rs_pool[:0]
            empty_coe = torch.zeros(1, 0, device=self.device)
            if not return_retr_details:
                return empty_c, empty_r, empty_coe
            return {
                'vis_sim': torch.zeros(1, 0, device=self.device),
                'vis_sim_prot': torch.zeros(1, 1, device=self.device),
                'retrieved_requests': [],
                'moe_cs': empty_c, 'moe_rs': empty_r, 'fuse_coe': empty_coe
            }
        
        # ==================== Feature Extraction ====================
        ivr = self.inpt_extractor.extract_vision(query_reps, vision_reps)  # [1, eqe_n, module_dim]
        iqr = self.inpt_extractor.extract_query(query_reps)                # [1, eqe_n, module_dim]
        
        # vis_sim for all experts
        vis_sim_all = torch.einsum('bed,med->bme', ivr, self.evr_pool).mean(2) * self.sim_scale  # [1, m]
        vis_sim_all = vis_sim_all[0]  # [m]
        
        # vis_sim_prot (for compatibility / debug)
        ivr_prot = self.inpt_extractor.extract_from_visprot(query_reps)  # [1, eqe_n, module_dim]
        vis_sim_prot = torch.einsum('bed,bed->be', ivr, ivr_prot).mean(1, True) * self.sim_scale  # [1, 1]
        
        # ==================== Routing Path Selection ====================
        if not cfg.use_joint_routing:
            # ==================== Legacy Hard+Soft Routing ====================
            return self._retrieve_moes_legacy(
                vision_reps, query_reps, ivr, iqr, vis_sim_all, vis_sim_prot, return_retr_details)
        
        # ==================== Joint Sparse Routing v2 ====================
        # Core idea: preserve ALL trained gating mechanisms from the legacy method
        # (hard gate via vis_sim_prot, soft abs_sim from sigmoid) as the baseline;
        # joint routing only applies MULTIPLICATIVE refinement on top.
        # This avoids destroying locality suppression and keeps reliability stable.
        
        s_vis_all = vis_sim_all  # [m]
        vis_sim_prot_scalar = vis_sim_prot.squeeze()  # scalar
        
        # ---------- Step 1: Hard gate (preserve locality rejection) ----------
        if cfg.routing_keep_hard_gate:
            hard_gate = (s_vis_all > vis_sim_prot_scalar)  # [m]
            if not hard_gate.any():
                # Truly no relevant expert (e.g. locality data) — skip editing
                empty_c = self.moe_cs_pool[:0]
                empty_r = self.moe_rs_pool[:0]
                empty_coe = torch.zeros(1, 0, device=self.device)
                if not return_retr_details:
                    return empty_c, empty_r, empty_coe
                return self._pack_retr_details(vis_sim_all.unsqueeze(0), vis_sim_prot, [],
                                               empty_c, empty_r, empty_coe,
                                               None, None, None, None, None, None)
        else:
            hard_gate = torch.ones(m_total, dtype=torch.bool, device=self.device)
        
        # ---------- Step 2: Preselection on gated experts ----------
        # Use vis_sim (main signal) + small quality boost for tie-breaking
        if len(self.influence_pool) > 0:
            s_qua_all = self._compute_quality_score(self.influence_pool, self.stability_pool)  # [m]
        else:
            s_qua_all = torch.zeros(m_total, device=self.device)
        
        # Preselection score: mainly vis_sim (trained signal), tiny quality boost
        s_pre_all = s_vis_all + 0.1 * (s_qua_all - 0.5)  # small centered boost
        # Mask out non-gated experts
        s_pre_masked = s_pre_all.masked_fill(~hard_gate, float('-inf'))
        
        n_gated = int(hard_gate.sum().item())
        preselect_k = min(cfg.preselect_k, n_gated)
        _, candidate_idx = s_pre_masked.topk(preselect_k)  # [K0]
        
        k = len(candidate_idx)
        
        # ---------- Step 3: Compute BASELINE fuse_coe using the trained legacy method ----------
        # This is exactly what the original soft routing produced (rela_sim * abs_sim).
        # All soft_routing_loss training is directly leveraged here.
        eqr_cand = self.eqr_pool[candidate_idx]  # [k, eqe_n, module_dim]
        rela_sim, abs_sim = self.get_moe_fuse_coe(iqr, eqr_cand, split=True)  # [1, k], [1, k]
        if cfg.routing_keep_abs_sim:
            base_fuse = rela_sim * abs_sim  # [1, k] — identical to legacy soft fusion
        else:
            base_fuse = rela_sim  # [1, k] — softmax only (not recommended)
        
        # ---------- Step 4: Joint refinement (MULTIPLICATIVE, small deviation) ----------
        # Centered boosts ensure neutral quality/mask leaves base_fuse unchanged.
        s_vis = s_vis_all[candidate_idx]  # [k]
        s_qua = s_qua_all[candidate_idx]  # [k]
        s_qry = self.get_query_sim_score(iqr, eqr_cand)  # [k]  (for logging only)
        
        fuse_coe = base_fuse  # [1, k]
        
        fuse_coe = refine_routing_weights(base_fuse,
            s_vis.unsqueeze(0), s_qry.unsqueeze(0), s_qua.unsqueeze(0),
            self.joint_refine_head if cfg.use_joint_head else None,
            cfg.routing_lambda_qua, cfg.joint_head_beta)

        # Mask boost (only when explicitly enabled; default off because infer-time ans_reps=None)
        s_mask = torch.zeros(k, device=self.device)
        if cfg.routing_lambda_mask > 0:
            infer_fine_mask, infer_coarse_score, infer_necessity_score = \
                self._compute_infer_local_signals(vision_reps, query_reps)
            s_mask = self._compute_mask_consistency(
                infer_fine_mask[0], infer_coarse_score[0], infer_necessity_score[0],
                candidate_idx)  # [k], in [-1, 1]
            mask_boost = (1.0 + cfg.routing_lambda_mask * s_mask).clamp(min=0.0, max=2.0)  # [k]
            fuse_coe = fuse_coe * mask_boost.unsqueeze(0)
        
        # ---------- Step 5: Optional thresholding (default OFF to preserve soft behavior) ----------
        if cfg.routing_min_weight > 0:
            active_mask = (fuse_coe[0] >= cfg.routing_min_weight)
            if active_mask.any():
                fuse_coe = fuse_coe * active_mask.float().unsqueeze(0)
        
        # NOTE: We deliberately do NOT re-normalize fuse_coe to sum=1.
        # Keeping the absolute scale preserves legacy abs_sim's gating behavior.
        
        # ---------- Step 6: Prune zero-weight experts (for efficiency) ----------
        active_mask = (fuse_coe[0] > 1e-8)
        if active_mask.any() and int(active_mask.sum().item()) < k:
            active_idx = active_mask.nonzero(as_tuple=True)[0]
            candidate_idx = candidate_idx[active_idx]
            fuse_coe = fuse_coe[:, active_idx]
            s_vis = s_vis[active_idx]
            s_qry = s_qry[active_idx]
            s_qua = s_qua[active_idx]
            s_mask = s_mask[active_idx]
        
        moe_cs = self.moe_cs_pool[candidate_idx]  # [k', rank, dim]
        moe_rs = self.moe_rs_pool[candidate_idx]  # [k', rank, dim]
        
        if not return_retr_details:
            return moe_cs, moe_rs, fuse_coe
        
        retrieved_requests = [self.requests_pool[i] for i in candidate_idx.tolist()]
        s_joint = fuse_coe[0]  # final combined weight for logging
        return self._pack_retr_details(
            vis_sim_all.unsqueeze(0), vis_sim_prot, retrieved_requests, 
            moe_cs, moe_rs, fuse_coe,
            candidate_idx, s_vis, s_qry, s_qua, s_mask, s_joint)
    
    def _compute_mask_consistency(self, infer_fine_mask: torch.Tensor,
                                   infer_coarse_score: torch.Tensor,
                                   infer_necessity_score: torch.Tensor,
                                   candidate_idx: torch.Tensor) -> torch.Tensor:
        """Compute mask consistency scores between input and pool experts.
        
        Args:
            infer_fine_mask: [lv] - current input's fine mask
            infer_coarse_score: [lv] - current input's coarse score
            infer_necessity_score: [lv] - current input's necessity score
            candidate_idx: [k] - indices of candidate experts
        Returns:
            s_mask: [k] - mask consistency scores
        """
        cfg = self.cfg.retrieval_editor
        k = len(candidate_idx)
        
        # vis_mask_pool: [m, lv]
        pool_fine_mask = self.vis_mask_pool[candidate_idx]  # [k, lv]
        
        # Primary: fine_mask consistency
        s_mask = self._safe_cosine_score(infer_fine_mask, pool_fine_mask)  # [k]
        
        count = 1.0
        # Optional: coarse_score consistency
        if cfg.routing_mask_use_coarse and len(self.coarse_score_pool) > 0:
            pool_coarse = self.coarse_score_pool[candidate_idx]  # [k, lv]
            s_coarse = self._safe_cosine_score(infer_coarse_score, pool_coarse)  # [k]
            s_mask = s_mask + s_coarse
            count += 1.0
        
        # Optional: necessity consistency
        if cfg.routing_mask_use_necessity and len(self.necessity_pool) > 0:
            pool_necessity = self.necessity_pool[candidate_idx]  # [k, lv]
            s_necessity = self._safe_cosine_score(infer_necessity_score, pool_necessity)  # [k]
            s_mask = s_mask + s_necessity
            count += 1.0
        
        return s_mask / count  # [k], averaged
    
    def _retrieve_moes_legacy(self, vision_reps: torch.Tensor, query_reps: torch.Tensor,
                              ivr: torch.Tensor, iqr: torch.Tensor,
                              vis_sim_all: torch.Tensor, vis_sim_prot: torch.Tensor,
                              return_retr_details: bool):
        """Legacy hard+soft routing (fallback when use_joint_routing=False)."""
        cfg = self.cfg.retrieval_editor
        vis_sim = vis_sim_all.unsqueeze(0)  # [1, m]
        
        # Stage 1: coarse hard routing
        retrieval_map = (vis_sim > vis_sim_prot).to(torch.bool)[0]  # [m]
        
        # Stage 2: quality-based shortlist
        if retrieval_map.any() and len(self.influence_pool) > 0:
            inf_thr = cfg.influence_threshold
            stab_thr = cfg.stability_threshold
            quality_ok = (self.influence_pool >= inf_thr) & (self.stability_pool >= stab_thr)
            refined_map = retrieval_map & quality_ok
            if refined_map.any():
                retrieval_map = refined_map
        
        # Stage 3: top-k shortlist
        shortlist_k = cfg.shortlist_k
        if shortlist_k > 0 and retrieval_map.sum() > shortlist_k:
            candidate_sims = vis_sim[0][retrieval_map]
            _, topk_local = candidate_sims.topk(shortlist_k)
            candidate_idx = retrieval_map.nonzero(as_tuple=True)[0]
            new_map = torch.zeros_like(retrieval_map)
            new_map[candidate_idx[topk_local]] = True
            retrieval_map = new_map
        
        # Fallback: empty retrieval
        if not retrieval_map.any():
            empty_c = self.moe_cs_pool[:0]
            empty_r = self.moe_rs_pool[:0]
            empty_coe = torch.zeros(1, 0, device=self.device)
            if not return_retr_details:
                return empty_c, empty_r, empty_coe
            return vis_sim, vis_sim_prot, [], empty_c, empty_r, empty_coe
        
        moe_cs = self.moe_cs_pool[retrieval_map]
        moe_rs = self.moe_rs_pool[retrieval_map]
        eqr = self.eqr_pool[retrieval_map]
        fuse_coe = self.get_moe_fuse_coe(iqr, eqr)
        
        if not return_retr_details:
            return moe_cs, moe_rs, fuse_coe
        retrieved_requests = [r for r, m in zip(self.requests_pool, retrieval_map) if m]
        return vis_sim, vis_sim_prot, retrieved_requests, moe_cs, moe_rs, fuse_coe
    
    def _pack_retr_details(self, vis_sim, vis_sim_prot, retrieved_requests, 
                           moe_cs, moe_rs, fuse_coe,
                           candidate_idx=None, s_vis=None, s_qry=None, 
                           s_qua=None, s_mask=None, s_joint=None):
        """Pack retrieval details for return_retr_details=True."""
        return {
            'vis_sim': vis_sim,
            'vis_sim_prot': vis_sim_prot,
            'retrieved_requests': retrieved_requests,
            'moe_cs': moe_cs,
            'moe_rs': moe_rs,
            'fuse_coe': fuse_coe,
            # Joint routing specific
            'candidate_idx': candidate_idx,
            's_vis': s_vis,
            's_qry': s_qry,
            's_qua': s_qua,
            's_mask': s_mask,
            's_joint': s_joint,
        }
    
    ############################################################################
    ######################### EGDER Basic Functions ############################
    ############################################################################
    def get_reps_for_edit(self, vllm:BaseVLLMForEdit, request: Dict):
        '''request: {'prompt': str, 'image': PILImage, 'target': str}'''
        (input_embeds, vt_range), label_ids, label_masks = vllm.prompts_imgs_target_to_xym(
            [request['prompt']], [request['image']], [request['target']])
        edit_signal = vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path) 
        pre_vision_reps = edit_signal[:, :vt_range[0]] # [1, l1, d]
        vision_reps = edit_signal[:, vt_range[0]:vt_range[1]] # [1, l1, d]
        query_reps = edit_signal[:, vt_range[1]:1 - label_masks.shape[1]] # [1, l2, d]
        ans_reps = edit_signal[:, 1 - label_masks.shape[1]:] # [1, l3, d]
        return pre_vision_reps, vision_reps, query_reps, ans_reps # Edit signals

    def get_new_edit(self, vision_reps:torch.Tensor, query_reps:torch.Tensor,
                     ans_reps:torch.Tensor) -> Tuple:
        # vision_reps: [1, lv, d]; query_reps: [1, lq, d]; ans_reps: [1, la, d]
        assert len(vision_reps) == len(query_reps) == len(ans_reps) == 1
        evr = self.edit_extractor.extract_vision(query_reps, vision_reps)
        eqr = self.edit_extractor.extract_query(query_reps)

        # Step 1: localize — coarse-to-fine mask on vision tokens
        mask_out = self.mask_localizer(vision_reps, query_reps, ans_reps)
        fine_mask = mask_out['fine_mask']               # [1, lv]
        coarse_score = mask_out['coarse_score']         # [1, lv]
        necessity_score = mask_out['necessity_score']   # [1, lv]

        # Step 2: construct masked edit reps (edit only in attribution region)
        masked_vision = vision_reps * fine_mask.unsqueeze(-1)   # [1, lv, d]
        masked_edit_reps = torch.cat([masked_vision, query_reps, ans_reps], 1)

        # Step 3: dual expert generation on masked region
        support_c, support_r, suppress_c, suppress_r = self.dual_moegen(masked_edit_reps)

        # Step 4: purity scoring
        magnitude = self.magnitude_head(masked_edit_reps)    # [1]
        quality_reps = torch.cat([vision_reps, query_reps], 1)
        influence_logits = self.influence_head.logits(quality_reps)
        influence = torch.sigmoid(influence_logits)     # [1]
        stability_logits = self.stability_head.logits(quality_reps)
        stability = torch.sigmoid(stability_logits)     # [1]

        # Step 5: compose equivalent residual expert (backward compatible)
        moe_c, moe_r = self.equiv_composer.compose_to_single(
            support_c, support_r, suppress_c, suppress_r, magnitude)

        meta = {
            'fine_mask': fine_mask, 'fine_logits': mask_out['fine_logits'], 'coarse_score': coarse_score,
            'necessity_logits': mask_out['necessity_logits'],
            'necessity_score': necessity_score,
            'support_c': support_c, 'support_r': support_r,
            'suppress_c': suppress_c, 'suppress_r': suppress_r,
            'magnitude': magnitude, 'influence': influence, 'stability': stability,
            'influence_logits': influence_logits, 'stability_logits': stability_logits,
        }
        return eqr, evr, moe_c, moe_r, meta
    
    def get_edit_residual(self, inpt_reps:torch.Tensor, moe_cs:torch.Tensor, 
                          moe_rs:torch.Tensor, fuse_coe:torch.Tensor)->torch.Tensor:
        '''inpt_reps: [1, l, hidden_size]; moe_cs: [m, lora_rank, hidden_size]; 
            moe_rs: [m, lora_rank, hidden_size]; fuse_coe: [1, m]'''
        assert len(inpt_reps) == len(fuse_coe) == 1
        if moe_cs.shape[0] == 0:
            return torch.zeros_like(inpt_reps)
        x = self.instant_reps_norm(inpt_reps)[0] # [1, l, hidden_size] -> [l, hidden_size]
        # x = torch.einsum('ld,mrd,mrD,m->lD', x, moe_cs, moe_rs, fuse_coe[0])
        x = torch.relu(torch.einsum('ld,mrd->lmr', x, moe_cs))
        x = torch.einsum('lmr,mrd,m->ld', x, moe_rs, fuse_coe[0])
        return x.unsqueeze(0) # [1, l, hidden_size]
    
    def get_moe_fuse_coe(self, iqrs:torch.Tensor, eqrs:torch.Tensor, split = False):
        sim = torch.einsum('ned,med->nme', iqrs, eqrs).mean(2) * self.sim_scale # [n, m]
        rela_sim = torch.softmax(sim, 1) # [n, m]
        abs_sim = torch.sigmoid(sim) # [n, m] rescale ???
        if split: 
            return rela_sim, abs_sim
        return rela_sim * abs_sim # [n, m]

    def get_query_sim_score(self, iqrs: torch.Tensor, eqrs: torch.Tensor) -> torch.Tensor:
        """Extract raw query similarity score (before rela*abs fusion).
        
        Args:
            iqrs: [1, eqe_n, module_dim] - input query representation
            eqrs: [m, eqe_n, module_dim] - edit query representations from pool
        Returns:
            sim: [m] - raw cosine similarity scores (scaled)
        """
        sim = torch.einsum('ned,med->nme', iqrs, eqrs).mean(2) * self.sim_scale  # [1, m]
        return sim[0]  # [m]

    ############################################################################
    #################### Joint Sparse Routing Helpers ##########################
    ############################################################################
    
    def _safe_cosine_score(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute cosine similarity between x and y.
        
        Args:
            x: [1, d] or [d] - single vector
            y: [m, d] - multiple vectors
        Returns:
            sim: [m] - cosine similarities
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, d]
        if x.dim() == 2 and x.shape[0] == 1:
            x = x.expand(y.shape[0], -1)  # [m, d]
        x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)
        y_norm = y / (y.norm(dim=-1, keepdim=True) + eps)
        sim = (x_norm * y_norm).sum(dim=-1)  # [m]
        return sim
    
    def _compute_quality_score(self, influence: torch.Tensor, stability: torch.Tensor, 
                               eps: float = 1e-8) -> torch.Tensor:
        """Compute continuous quality score from influence and stability.
        
        Args:
            influence: [m] - influence scores for candidate experts
            stability: [m] - stability scores for candidate experts
        Returns:
            score: [m] - combined quality scores
        """
        if self.cfg.retrieval_editor.routing_quality_log:
            score = 0.5 * (torch.log(influence + eps) + torch.log(stability + eps))
            # Normalize to roughly [0,1] range: log scores typically in [-8, 0] for (0,1) inputs
            score = (score + 8.0) / 8.0  # shift and scale
            score = score.clamp(0.0, 1.0)
        else:
            score = 0.5 * (influence + stability)
        return score
    
    def _sparse_normalize(self, scores: torch.Tensor, tau: float, min_weight: float,
                          use_entmax: bool = False) -> torch.Tensor:
        """Sparse normalization of routing scores to get final weights.
        
        Args:
            scores: [k] or [1, k] - joint routing scores
            tau: temperature for softmax
            min_weight: threshold below which weights are zeroed
            use_entmax: if True, use entmax (requires entmax package); else softmax+threshold
        Returns:
            weights: [1, k] - sparse normalized weights
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)  # [1, k]
        
        k = scores.shape[1]
        if k == 0:
            return scores  # empty tensor, return as-is
        
        if k == 1:
            return torch.ones_like(scores)  # single expert, weight = 1.0
        
        if use_entmax:
            # Try entmax if available; fallback to softmax+threshold
            try:
                from entmax import entmax15
                weights = entmax15(scores / tau, dim=-1)  # [1, k]
            except ImportError:
                # Fallback to softmax + threshold
                weights = torch.softmax(scores / tau, dim=-1)  # [1, k]
                weights = weights * (weights >= min_weight).float()
        else:
            # Default: softmax + threshold
            weights = torch.softmax(scores / tau, dim=-1)  # [1, k]
            weights = weights * (weights >= min_weight).float()
        
        # Re-normalize if any weights remain
        weight_sum = weights.sum(dim=-1, keepdim=True)
        if weight_sum.item() > 1e-8:
            weights = weights / weight_sum
        else:
            # All weights zeroed: fallback to argmax
            max_idx = scores.argmax(dim=-1)
            weights = torch.zeros_like(scores)
            weights[0, max_idx] = 1.0
        
        return weights  # [1, k]
    
    def _compute_infer_local_signals(self, vision_reps: torch.Tensor, 
                                      query_reps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute local signals for inference retrieval (without ans_reps).
        
        Used only in retrieve_moes() for mask consistency scoring.
        
        Args:
            vision_reps: [1, lv, d]
            query_reps: [1, lq, d]
        Returns:
            fine_mask: [1, lv]
            coarse_score: [1, lv]
            necessity_score: [1, lv]
        """
        with torch.no_grad():
            mask_out = self.mask_localizer(vision_reps, query_reps, ans_reps=None)
        return mask_out['fine_mask'], mask_out['coarse_score'], mask_out['necessity_score']

    ############################################################################
    ########################### Editor Functions ###############################
    ############################################################################
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'egder', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return False

    def restore_to_original_model(self):
        self.requests_pool = []
        # Original pools
        self.eqr_pool = torch.zeros([0, self.cfg.retrieval_editor.eqe_n, self.cfg.retrieval_editor.module_dim], device=self.device)
        self.evr_pool = torch.zeros([0, self.cfg.retrieval_editor.eqe_n, self.cfg.retrieval_editor.module_dim], device=self.device)
        self.moe_cs_pool = torch.zeros([0, self.cfg.retrieval_editor.lora_rank, self.cfg.llm_mid_dim], device=self.device)
        self.moe_rs_pool = torch.zeros([0, self.cfg.retrieval_editor.lora_rank, self.cfg.llm_mid_dim], device=self.device)
        # New pools for localized editing memory
        lora_rank = self.cfg.retrieval_editor.lora_rank
        mid_dim = self.cfg.llm_mid_dim
        self.vis_mask_pool = torch.zeros([0, self.vision_tok_n], device=self.device)
        self.necessity_pool = torch.zeros([0, self.vision_tok_n], device=self.device)
        self.coarse_score_pool = torch.zeros([0, self.vision_tok_n], device=self.device)
        self.influence_pool = torch.zeros([0], device=self.device)
        self.stability_pool = torch.zeros([0], device=self.device)
        self.magnitude_pool = torch.zeros([0], device=self.device)
        self.support_cs_pool = torch.zeros([0, lora_rank, mid_dim], device=self.device)
        self.support_rs_pool = torch.zeros([0, lora_rank, mid_dim], device=self.device)
        self.suppress_cs_pool = torch.zeros([0, lora_rank, mid_dim], device=self.device)
        self.suppress_rs_pool = torch.zeros([0, lora_rank, mid_dim], device=self.device)

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """
        self.is_editing = True
        self.requests_pool.append(request)
        r = {'prompt': request['prompt'], 'image': request['image'], 'target': request['target_new']}
        pre_vision_reps, vision_reps, query_reps, ans_reps = self.get_reps_for_edit(self.vllm, r)
        eqr, evr, moe_c, moe_r, meta = self.get_new_edit(vision_reps, query_reps, ans_reps)

        # Expert purity filtering: degrade low-quality experts gracefully
        inf_s = meta['influence'].item()
        stab_s = meta['stability'].item()
        if (inf_s < self.cfg.retrieval_editor.influence_threshold or
                stab_s < self.cfg.retrieval_editor.stability_threshold):
            degrade = max(inf_s * stab_s, 0.01)
            moe_c = moe_c * degrade
            moe_r = moe_r * degrade

        # Legacy pool writes (backward compatible)
        self.eqr_pool = torch.cat([self.eqr_pool, eqr], 0)
        self.evr_pool = torch.cat([self.evr_pool, evr], 0)
        self.moe_cs_pool = torch.cat([self.moe_cs_pool, moe_c], 0)
        self.moe_rs_pool = torch.cat([self.moe_rs_pool, moe_r], 0)
        # New localized editing pool writes
        self.vis_mask_pool = torch.cat([self.vis_mask_pool, meta['fine_mask']], 0)
        self.necessity_pool = torch.cat([self.necessity_pool, meta['necessity_score']], 0)
        self.coarse_score_pool = torch.cat([self.coarse_score_pool, meta['coarse_score']], 0)
        self.influence_pool = torch.cat([self.influence_pool, meta['influence']])
        self.stability_pool = torch.cat([self.stability_pool, meta['stability']])
        self.magnitude_pool = torch.cat([self.magnitude_pool, meta['magnitude']])
        self.support_cs_pool = torch.cat([self.support_cs_pool, meta['support_c']], 0)
        self.support_rs_pool = torch.cat([self.support_rs_pool, meta['support_r']], 0)
        self.suppress_cs_pool = torch.cat([self.suppress_cs_pool, meta['suppress_c']], 0)
        self.suppress_rs_pool = torch.cat([self.suppress_rs_pool, meta['suppress_r']], 0)
        self.is_editing = False 

    def edit_batch(self, requests: List[Dict]):
        raise
    
    ############################################################################
    ####################### Training Functions #################################
    ############################################################################
    def get_modules_for_training(self)->Dict[str, nn.Module]:
        return {'edit_extractor': self.edit_extractor, 'inpt_extractor': self.inpt_extractor, 
            'moegen_c': self.moegen_c, 'moegen_r': self.moegen_r, 
            'instant_reps_norm': self.instant_reps_norm,
            'mask_localizer': self.mask_localizer,
            'dual_moegen': self.dual_moegen,
            'magnitude_head': self.magnitude_head,
            'influence_head': self.influence_head,
            'stability_head': self.stability_head,
            'equiv_composer': self.equiv_composer,
            'joint_refine_head': self.joint_refine_head}
    
    def reinit_train_parameters(self):
        self.edit_extractor.reset_parameters()
        self.inpt_extractor.reset_parameters()
        self.moegen_c.reset_parameters()
        self.moegen_r.reset_parameters()
        self.instant_reps_norm.reset_parameters()
        self.mask_localizer.reset_parameters()
        self.dual_moegen.reset_parameters()
        self.magnitude_head.reset_parameters()
        self.influence_head.reset_parameters()
        self.stability_head.reset_parameters()
        self.equiv_composer.reset_parameters()
        self.joint_refine_head.reset_parameters()

    def preprocess_train_data(self, vllm_edit_data:BaseVLLMEditData)->List:
        return vllm_edit_data.data_with_img
        
        
    def organize_batch_data(self, a_batch_raw_data:List):
        vllm = self.vllm_data_proc # use the VLLM used to preprocess data
        batch_size = len(a_batch_raw_data)
        batch_edit_signal = []
        rel_edit_i = [] 
        batch_rel_data = [] # for each: {'prompts': [str], 'imgs': [PILImage], 'targets': [str]}
        batch_gen_data = {gen_name: [] for gen_name in a_batch_raw_data[0]['generality']}
        batch_loc_data = {loc_name: [] for loc_name in a_batch_raw_data[0]['locality']
                if a_batch_raw_data[0]['locality'][loc_name][0]['image'] != None}
        for d in a_batch_raw_data:
            # Edit signal
            batch_edit_signal.append([self.get_reps_for_edit(vllm, {'prompt': r['prompt'],
                'image': r['image'], 'target': r['target_new']}) for r in d['requests']])
            # Reliability 
            i = self.rng_data_proc.integers(0, len(d['requests']))
            rel_edit_i.append(i)
            r = d['requests'][i]
            batch_rel_data.append({'prompts': [r['prompt']], 'imgs': [r['image']], 'targets': [r['target_new']]})
            # Generality
            for gen_name in batch_gen_data.keys():
                i = self.rng_data_proc.integers(0, len(d['generality'][gen_name]))
                g = d['generality'][gen_name][i]
                batch_gen_data[gen_name].append({'prompts': [g['prompt']], 'imgs': [g['image']], 'targets': [g['target']]})
            # Locality
            for loc_name in batch_loc_data.keys():
                i = self.rng_data_proc.integers(0, len(d['locality'][loc_name]))
                l = d['locality'][loc_name][i]
                batch_loc_data[loc_name].append({'prompts': [l['prompt']], 'imgs': [l['image']], 'targets': [l['target']]})
        # Fuse moe masks
        edit_ns = torch.tensor([len(bes) for bes in batch_edit_signal], device=self.device)
        cols = torch.sum(edit_ns)
        start_indices = torch.cumsum(torch.cat([torch.tensor([0], device=self.device), edit_ns[:-1]]), dim=0).unsqueeze(1)
        rel_edit_i = start_indices[:, 0] + torch.tensor(rel_edit_i, device=self.device)
        rel_moe_mask = torch.zeros((batch_size, cols), dtype=torch.int, device=self.device)
        rel_moe_mask[torch.arange(batch_size), rel_edit_i] = 1
        idx = torch.arange(cols, device = self.device).expand(batch_size, cols)
        gen_moe_mask = ((idx >= start_indices) & (idx < (start_indices + edit_ns.unsqueeze(1)))).int()
        loc_moe_mask = torch.zeros_like(gen_moe_mask) 
        for i in range(batch_size):
            ns = self.rng_train.integers(0, int(cols) + 1, 3)
            rel_moe_mask[i, :ns[0]], gen_moe_mask[i, :ns[1]], loc_moe_mask[i, :ns[2]] = 1, 1, 1
        rel_moe_mask, gen_moe_mask, loc_moe_mask = rel_moe_mask.bool(), gen_moe_mask.bool(), loc_moe_mask.bool()
        # Reliability edit batch data
        rel_xyms = [vllm.prompts_imgs_target_to_xym(**r) for r in batch_rel_data] # for each: (input_embeds, vt_range), label_ids, label_masks
        rel_mid_reps = [vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path) 
                        for (input_embeds, vt_range), label_ids, label_masks in rel_xyms] 
        rel_edit_reps = [self.get_reps_for_edit(vllm, {'prompt': r['prompts'][0], 
            'image': r['imgs'][0], 'target': r['targets'][0]}) for r in batch_rel_data]
        packed_rel_data = (rel_xyms, rel_mid_reps, rel_edit_reps)
        # Generality edit batch data
        packed_gen_data = {}
        for k, v in batch_gen_data.items():
            gen_xyms = [vllm.prompts_imgs_target_to_xym(**g) for g in v]
            gen_mid_reps = [vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path) 
                for (input_embeds, vt_range), label_ids, label_masks in gen_xyms] 
            gen_edit_reps = [self.get_reps_for_edit(vllm, {'prompt': g['prompts'][0], 
                'image': g['imgs'][0], 'target': g['targets'][0]}) for g in v]
            packed_gen_data[k] = (gen_xyms, gen_mid_reps, gen_edit_reps)
        # Locality edit batch data
        packed_loc_data = {}
        for k, v in batch_loc_data.items():
            loc_xyms = [vllm.prompts_imgs_target_to_xym(**l) for l in v]
            loc_mid_reps = [vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path) 
                for (input_embeds, vt_range), label_ids, label_masks in loc_xyms] 
            pre_logits = [vllm.forward_from_mid_layer(input_embeds, vt_range, lmr, 
                self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits for 
                lmr, ((input_embeds, vt_range), label_ids, label_masks) in zip(loc_mid_reps, loc_xyms)]
            loc_edit_reps = [self.get_reps_for_edit(vllm, {'prompt': l['prompts'][0], 
                'image': l['imgs'][0], 'target': l['targets'][0]}) for l in v] 
            packed_loc_data[k] = (loc_xyms, loc_mid_reps, pre_logits, loc_edit_reps)
        # Retrieval contrastive learning data for neighbor loss
        def get_rand_gn_ln():
            gn = list(batch_gen_data.keys())[self.rng_data_proc.integers(0, len(batch_gen_data.keys()))]
            ln = list(batch_loc_data.keys())[self.rng_data_proc.integers(0, len(batch_loc_data.keys()))]
            return gn, ln
        batch_retr_neib_data = [[], []]
        for j in range(batch_size):
            # neighbor 1
            n = self.rng_data_proc.integers(0, 3) # 0: rel, 1: gen, 2: loc. Where rel and gen are in the same neighbor.
            gn, ln = get_rand_gn_ln()
            d = [packed_rel_data[-1], packed_gen_data[gn][-1], packed_loc_data[ln][-1]][n][j][1:3]
            batch_retr_neib_data[0].append(d)
            # neighbor 2 
            n = self.rng_data_proc.integers(0, 2) if n != 2 else n # reselect in the same neighbor
            gn, ln = get_rand_gn_ln()
            d = [packed_rel_data[-1], packed_gen_data[gn][-1], packed_loc_data[ln][-1]][n][j][1:3]
            batch_retr_neib_data[1].append(d)
        # Retrieval contrastive learning data for prototype loss 
        batch_retr_prot_data = [[], []]
        for j in range(batch_size):
            n = self.rng_data_proc.integers(0, 2) # 0: r/g, 1: l
            # data_1
            gn, ln = get_rand_gn_ln()
            d = [[packed_rel_data[-1], packed_gen_data[gn][-1]][self.rng_data_proc.integers(0, 2)], packed_loc_data[ln][-1]][n][j][1:3]
            batch_retr_prot_data[0].append(d)
            # data_2
            gn, ln = get_rand_gn_ln()
            d = [[packed_rel_data[-1], packed_gen_data[gn][-1]][self.rng_data_proc.integers(0, 2)], packed_loc_data[ln][-1]][1-n][j][1:3]
            batch_retr_prot_data[1].append(d)
        a_batch_organized_data = (batch_size, batch_edit_signal, rel_moe_mask, gen_moe_mask, loc_moe_mask, 
            packed_rel_data, packed_gen_data, packed_loc_data, batch_retr_neib_data, batch_retr_prot_data)
        return move_to_device(a_batch_organized_data, self.device)
        
    def train_a_batch(self, a_batch_organized_data):
        eps = 1e-8
        vllm = self.vllm
        (batch_size, batch_edit_signal, rel_moe_mask, gen_moe_mask, loc_moe_mask, 
            packed_rel_data, packed_gen_data, packed_loc_data, batch_retr_neib_data, 
            batch_retr_prot_data) = a_batch_organized_data
        # initialize losses
        loss = 0
        log_dict = {}
        # Prepare edit data: eqr, evr, moe_c, moe_r, meta
        new_edit = [self.get_new_edit(vision_reps, query_reps, ans_reps) for bes in batch_edit_signal 
                    for pre_vision_reps, vision_reps, query_reps, ans_reps in bes]
        eqrs = torch.cat([ne[0] for ne in new_edit], 0) # [batch_request_n, eqe_n, d]
        # evrs = torch.cat([ne[1] for ne in new_edit], 0) # [batch_request_n, eqe_n, d]
        moe_cs = torch.cat([ne[2] for ne in new_edit], 0) # [batch_request_n, rank_n, d1]
        moe_rs = torch.cat([ne[3] for ne in new_edit], 0) # [batch_request_n, rank_n, d2]
        assert all(len(bes) == 1 for bes in batch_edit_signal), 'Repaired quality supervision requires one source request per EVQA case.'
        influence_targets = []
        stability_kl = torch.zeros(batch_size, device=self.device)
        stability_count = 0
        # Reliability edit loss
        rel_loss = 0
        rel_xyms, rel_mid_reps, rel_edit_reps = packed_rel_data
        for xyms, mr, er, mm in zip(rel_xyms, rel_mid_reps, rel_edit_reps, rel_moe_mask):
            (input_embeds, vt_range), label_ids, label_masks = xyms
            pre_vision_reps, vision_reps, query_reps, ans_reps = er
            iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
            fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm]) # [1, selected_request_n]
            self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), 
                moe_cs[mm], moe_rs[mm], fuse_coe)
            logits = vllm.forward_from_mid_layer(input_embeds, vt_range, 
                mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
            source_loss = vllm.label_loss(logits, label_ids, label_masks, True)
            rel_loss += source_loss
            influence_targets.append(torch.exp(-source_loss.detach()).clamp(0., 1.))
        rel_loss /= batch_size
        log_dict['Reliability loss'] = float(rel_loss)
        loss += rel_loss * self.cfg.train_cfg.rel_lambda
        # Generality edit losses
        gen_loss = 0
        for gen_name, (gen_xyms, gen_mid_reps, gen_edit_reps) in packed_gen_data.items():
            gen_name_loss = 0
            for xyms, mr, er, mm in zip(gen_xyms, gen_mid_reps, gen_edit_reps, gen_moe_mask):
                (input_embeds, vt_range), label_ids, label_masks = xyms
                pre_vision_reps, vision_reps, query_reps, ans_reps = er
                iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
                fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm]) # [1, selected_request_n]
                self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), 
                    moe_cs[mm], moe_rs[mm], fuse_coe)
                logits = vllm.forward_from_mid_layer(input_embeds, vt_range, 
                    mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
                gen_name_loss += vllm.label_loss(logits, label_ids, label_masks, True)
            gen_name_loss /= batch_size
            log_dict['Generality loss %s'%gen_name] = float(gen_name_loss)
            gen_loss += gen_name_loss
        log_dict['Generality loss'] = float(gen_loss)
        loss += gen_loss * self.cfg.train_cfg.gen_lambda
        # Locality edit losses
        loc_loss = 0
        for loc_name, (loc_xyms, loc_mid_reps, pre_logits, loc_edit_reps) in packed_loc_data.items():
            loc_name_loss = 0
            for quality_i, (xyms, mr, pl, er, mm) in enumerate(zip(loc_xyms, loc_mid_reps, pre_logits, loc_edit_reps, loc_moe_mask)):
                (input_embeds, vt_range), label_ids, label_masks = xyms
                pre_vision_reps, vision_reps, query_reps, ans_reps = er
                iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
                fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm]) # [1, selected_request_n]
                self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), 
                    moe_cs[mm], moe_rs[mm], fuse_coe)
                logits = vllm.forward_from_mid_layer(input_embeds, vt_range, 
                    mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
                case_kl = vllm.logit_KL_loss(logits, pl, label_masks, True)
                loc_name_loss += case_kl
                stability_kl[quality_i] += case_kl.detach().clamp(min=0.)
            stability_count += 1
            loc_name_loss /= batch_size
            log_dict['Locality loss %s'%loc_name] = float(loc_name_loss)
            loc_loss += loc_name_loss
        log_dict['Locality loss'] = float(loc_loss)
        loss += loc_loss * self.cfg.train_cfg.loc_lambda
        # Moe soft routing loss
        iqrs = torch.cat([self.inpt_extractor.extract_query(d[1]) for d in batch_retr_neib_data[0]], 0) # [b, eqe_n, module_dim]
        eqrs = torch.cat([self.edit_extractor.extract_query(d[1]) for d in batch_retr_neib_data[1]], 0) # [b, eqe_n, module_dim]
        rela_sim, abs_sim = self.get_moe_fuse_coe(iqrs, eqrs, True) # [b, b]
        moe_soft_routing_loss_rela = - torch.log(torch.diag(rela_sim) + eps).mean(0)
        abs_pos_sim = torch.diag(abs_sim)
        abs_neg_sim = torch.diag(abs_sim.roll(1, 1))
        moe_soft_routing_loss_abs = - (torch.log(abs_pos_sim + eps) + torch.log(1 - abs_neg_sim + eps)).mean()
        moe_soft_routing_loss = moe_soft_routing_loss_rela + moe_soft_routing_loss_abs
        log_dict['MoE soft routing relative loss'] = float(moe_soft_routing_loss_rela)
        log_dict['MoE soft routing abstract loss'] = float(moe_soft_routing_loss_abs)
        log_dict['MoE soft routing mean abstract positive similarity'] = float(abs_pos_sim.mean(0)) 
        log_dict['MoE soft routing mean abstract negative similarity'] = float(abs_neg_sim.mean(0))
        log_dict['MoE soft routing loss'] = float(moe_soft_routing_loss)
        loss += moe_soft_routing_loss * self.cfg.train_cfg.soft_routing_lambda
        # Hard Routing losses
        def hard_routing(inpt_reps, edit_reps):
            ivrs = torch.cat([self.inpt_extractor.extract_vision(d[1], d[0]) for d in inpt_reps], 0) # [b, eqe_n, module_dim]
            evrs = torch.cat([self.edit_extractor.extract_vision(d[1], d[0]) for d in edit_reps], 0) # [b, eqe_n, module_dim]
            sim = torch.einsum('bed,med->bme', ivrs, evrs).mean(2) * self.sim_scale # [b, b]
            ivrs_prot = torch.cat([self.inpt_extractor.extract_from_visprot(d[1]) for d in inpt_reps]) # [b, eqe_n, module_dim]
            sim_prot = torch.einsum('bed,bed->be', ivrs, ivrs_prot).mean(1, True) * self.sim_scale # [b,1]
            sim = torch.softmax(torch.cat([sim, sim_prot], 1), 1) # [b, b+1]
            return sim # [b, b+1]
        # Hard routing neighbor loss
        sim = hard_routing(*batch_retr_neib_data) # [b, b+1]
        loss_retr_neb = - torch.log(torch.diag(sim) + eps).mean(0)
        log_dict['MoE hard routing neighbor loss'] = float(loss_retr_neb)
        # Hard routing prototype loss 
        sim = hard_routing(*batch_retr_prot_data) # [b, b+1]
        loss_retr_prot = - torch.log(sim[:, -1] + eps).mean(0)
        log_dict['MoE hard routing prototype loss'] = float(loss_retr_prot)
        hard_routing_loss = loss_retr_neb + loss_retr_prot 
        log_dict['MoE hard routing loss'] = float(hard_routing_loss)
        loss += hard_routing_loss * self.cfg.train_cfg.hard_routing_lambda


        def _joint_weights(inpt_reps, edit_reps):
            ivrs = torch.cat([self.inpt_extractor.extract_vision(d[1], d[0]) for d in inpt_reps], 0)
            iqrs = torch.cat([self.inpt_extractor.extract_query(d[1]) for d in inpt_reps], 0)
            evrs = torch.cat([self.edit_extractor.extract_vision(d[1], d[0]) for d in edit_reps], 0)
            eqrs_r = torch.cat([self.edit_extractor.extract_query(d[1]) for d in edit_reps], 0)
            s_vis = (torch.einsum('bed,med->bme', ivrs, evrs).mean(2) * self.sim_scale).detach()
            s_qry = (torch.einsum('bed,med->bme', iqrs, eqrs_r).mean(2) * self.sim_scale).detach()
            # Quality uses the same answer-free hidden states as edit-time metadata.
            quality = torch.cat([.5 * (self.influence_head(torch.cat(d, 1)) +
                                      self.stability_head(torch.cat(d, 1))) for d in edit_reps], 0).detach()
            s_qua = quality.unsqueeze(0).expand_as(s_qry)
            base_fuse = torch.softmax(s_qry, 1) * torch.sigmoid(s_qry)
            cfg = self.cfg.retrieval_editor
            return refine_routing_weights(base_fuse, s_vis, s_qry, s_qua,
                self.joint_refine_head if cfg.use_joint_head else None,
                cfg.routing_lambda_qua, cfg.joint_head_beta)

        # Brier supervision applies directly to the same, non-normalized fusion
        # weights used after visual candidate selection at inference. The original
        # hard-routing losses still train visual acceptance/null rejection.
        neighbor_weights = _joint_weights(*batch_retr_neib_data)
        prototype_weights = _joint_weights(*batch_retr_prot_data)
        desired = torch.eye(batch_size, device=self.device)
        joint_neib_loss = (neighbor_weights - desired).square().mean()
        joint_prot_loss = prototype_weights.square().mean()
        joint_routing_loss = joint_neib_loss + joint_prot_loss
        log_dict['Joint routing neighbor loss'] = float(joint_neib_loss)
        log_dict['Joint routing prototype loss'] = float(joint_prot_loss)
        log_dict['Joint routing loss'] = float(joint_routing_loss)
        loss += joint_routing_loss * self.cfg.train_cfg.joint_routing_lambda

        # === New: localization & expert purity losses ===
        all_meta = [ne[4] for ne in new_edit]
        all_fine_masks = torch.cat([m['fine_mask'] for m in all_meta], 0)       # [N, lv]
        all_necessity = torch.cat([m['necessity_score'] for m in all_meta], 0)  # [N, lv]
        all_influence = torch.cat([m['influence'] for m in all_meta], 0)        # [N]
        all_stability = torch.cat([m['stability'] for m in all_meta], 0)        # [N]
        all_magnitude = torch.cat([m['magnitude'] for m in all_meta], 0)        # [N]
        all_suppress_c = torch.cat([m['suppress_c'] for m in all_meta], 0)      # [N, r, d]
        all_suppress_r = torch.cat([m['suppress_r'] for m in all_meta], 0)      # [N, r, d]

        fine_logits = torch.cat([m['fine_logits'] for m in all_meta], 0)
        mask_loss = mask_regularization_loss(all_fine_masks, fine_logits)
        log_dict['Mask loss'] = float(mask_loss)
        log_dict['Mask raw logit absolute max'] = float(fine_logits.detach().abs().max())
        log_dict['Mask mean'] = float(all_fine_masks.detach().mean())
        log_dict['Mask spatial variance'] = float(all_fine_masks.detach().var(unbiased=False))
        loss += mask_loss * self.cfg.train_cfg.mask_lambda

        necessity_logits = torch.cat([m['necessity_logits'] for m in all_meta], 0)
        necessity_loss = quality_supervision_loss(necessity_logits, all_fine_masks)
        log_dict['Necessity loss'] = float(necessity_loss)
        loss += necessity_loss * self.cfg.train_cfg.necessity_lambda

        influence_logits = torch.cat([m['influence_logits'] for m in all_meta], 0)
        influence_target = torch.stack(influence_targets).reshape(-1)
        stability_logits = torch.cat([m['stability_logits'] for m in all_meta], 0)
        stability_target = torch.exp(-stability_kl / max(stability_count, 1))
        influence_loss = quality_supervision_loss(influence_logits, influence_target)
        stability_loss = quality_supervision_loss(stability_logits, stability_target)
        log_dict['Influence loss'] = float(influence_loss)
        log_dict['Stability loss'] = float(stability_loss)
        for name, target, prediction in (('Influence', influence_target, all_influence),
                                         ('Stability', stability_target, all_stability)):
            log_dict[name + ' target min'] = float(target.min())
            log_dict[name + ' target max'] = float(target.max())
            log_dict[name + ' prediction mean'] = float(prediction.detach().mean())
        loss += influence_loss * self.cfg.train_cfg.influence_lambda
        loss += stability_loss * self.cfg.train_cfg.stability_lambda

        # Suppress regularization: prevent suppression expert from dominating
        suppress_loss = (all_suppress_c.norm(dim=-1).mean() +
                         all_suppress_r.norm(dim=-1).mean()) * 0.5
        log_dict['Suppress loss'] = float(suppress_loss)
        loss += suppress_loss * self.cfg.train_cfg.suppress_lambda

        # Magnitude regularization
        magnitude_loss = all_magnitude.mean()
        log_dict['Magnitude loss'] = float(magnitude_loss)
        loss += magnitude_loss * self.cfg.train_cfg.magnitude_lambda

        # update
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.lr_scheduler.step()
        log_dict['Learning rate'] = float(self.lr_scheduler.get_last_lr()[0])
        return float(loss), log_dict
    
    def compute_joint_routing_scores_for_training(self, 
                                                   ivr: torch.Tensor,
                                                   iqr: torch.Tensor,
                                                   evr_pool: torch.Tensor,
                                                   eqr_pool: torch.Tensor,
                                                   influence_pool: torch.Tensor,
                                                   stability_pool: torch.Tensor,
                                                   infer_fine_mask: torch.Tensor,
                                                   infer_coarse_score: torch.Tensor,
                                                   infer_necessity_score: torch.Tensor,
                                                   vis_mask_pool: torch.Tensor,
                                                   coarse_score_pool: torch.Tensor,
                                                   necessity_pool: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Placeholder for training-time joint routing score computation.
        
        This function is reserved for future joint routing training loss.
        Currently, training uses the legacy hard/soft routing losses.
        
        Args:
            ivr: [b, eqe_n, module_dim] - input vision representations
            iqr: [b, eqe_n, module_dim] - input query representations
            evr_pool: [m, eqe_n, module_dim] - edit vision reps pool
            eqr_pool: [m, eqe_n, module_dim] - edit query reps pool
            influence_pool: [m] - influence scores
            stability_pool: [m] - stability scores
            infer_fine_mask: [b, lv] - inferred fine masks
            infer_coarse_score: [b, lv] - inferred coarse scores
            infer_necessity_score: [b, lv] - inferred necessity scores
            vis_mask_pool: [m, lv] - pooled fine masks
            coarse_score_pool: [m, lv] - pooled coarse scores
            necessity_pool: [m, lv] - pooled necessity scores
            
        Returns:
            Dict containing:
                s_vis: [b, m] - visual similarity scores
                s_qry: [b, m] - query similarity scores
                s_qua: [m] - quality scores
                s_mask: [b, m] - mask consistency scores
                s_joint: [b, m] - combined joint routing scores
        """
        cfg = self.cfg.retrieval_editor
        b = ivr.shape[0]
        m = evr_pool.shape[0]
        
        # s_vis: visual similarity (batched)
        s_vis = torch.einsum('bed,med->bme', ivr, evr_pool).mean(2) * self.sim_scale  # [b, m]
        
        # s_qry: query similarity (batched)
        s_qry = torch.einsum('bed,med->bme', iqr, eqr_pool).mean(2) * self.sim_scale  # [b, m]
        
        # s_qua: quality score (shared across batch)
        s_qua = self._compute_quality_score(influence_pool, stability_pool)  # [m]
        
        # s_mask: mask consistency (batched)
        # For each sample in batch, compute cosine with all pool entries
        s_mask = torch.zeros(b, m, device=ivr.device)
        for i in range(b):
            fine_mask_i = infer_fine_mask[i]  # [lv]
            s_mask_i = self._safe_cosine_score(fine_mask_i, vis_mask_pool)  # [m]
            count = 1.0
            if cfg.routing_mask_use_coarse and coarse_score_pool.shape[0] > 0:
                coarse_i = infer_coarse_score[i]  # [lv]
                s_coarse_i = self._safe_cosine_score(coarse_i, coarse_score_pool)  # [m]
                s_mask_i = s_mask_i + s_coarse_i
                count += 1.0
            if cfg.routing_mask_use_necessity and necessity_pool.shape[0] > 0:
                nec_i = infer_necessity_score[i]  # [lv]
                s_nec_i = self._safe_cosine_score(nec_i, necessity_pool)  # [m]
                s_mask_i = s_mask_i + s_nec_i
                count += 1.0
            s_mask[i] = s_mask_i / count
        
        # s_joint: combined score
        s_joint = (cfg.routing_lambda_vis * s_vis + 
                   cfg.routing_lambda_qry * s_qry + 
                   cfg.routing_lambda_qua * s_qua.unsqueeze(0) + 
                   cfg.routing_lambda_mask * s_mask)  # [b, m]
        
        return {
            's_vis': s_vis,
            's_qry': s_qry,
            's_qua': s_qua,
            's_mask': s_mask,
            's_joint': s_joint,
        }
    
    def get_a_new_optimizer(self)->torch.optim.Optimizer:
        from torch.optim.lr_scheduler import LambdaLR
        opt = Adam([*self.edit_extractor.parameters(), *self.inpt_extractor.parameters(),
                    *self.instant_reps_norm.parameters(), *self.moegen_c.parameters(), 
                    *self.moegen_r.parameters(),
                    *self.mask_localizer.parameters(),
                    *self.dual_moegen.parameters(),
                    *self.magnitude_head.parameters(),
                    *self.influence_head.parameters(),
                    *self.stability_head.parameters(),
                    *self.equiv_composer.parameters(),
                    *self.joint_refine_head.parameters()], self.cfg.train_cfg.lr)
        lr_cut_it = torch.tensor(self.cfg.train_cfg.lr_cut_it, device=self.device)
        def lr_lambda(step):
            return self.cfg.train_cfg.lr_cut_rate**(step > lr_cut_it).sum()
        lr_scheduler = LambdaLR(opt, lr_lambda)
        return opt, lr_scheduler

    def set_train(self, is_train:bool):
        self.is_train = is_train
        self.vllm.model.requires_grad_(False)
        self.vllm.model.train(False)
        if hasattr(self, 'vllm_data_proc') and self.vllm_data_proc != None:
            self.vllm_data_proc.model.requires_grad_(False)
            self.vllm_data_proc.model.train(False)
        for mod in (self.edit_extractor, self.inpt_extractor,
                    self.instant_reps_norm, self.moegen_c, self.moegen_r,
                    self.mask_localizer, self.dual_moegen,
                    self.magnitude_head, self.influence_head,
                    self.stability_head, self.equiv_composer,
                    self.joint_refine_head):
            mod.train(is_train)
            mod.requires_grad_(is_train)

    def other_train_init_begin(self):
        self.rng_data_proc = np.random.default_rng(self.random_seed)
        self.rng_train = np.random.default_rng(self.random_seed + 1)
