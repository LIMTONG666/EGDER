# Attention, QVExtractor, and LowRankGenerator build on LiveEdit (Chen et al., CVPR 2025).
# EGDER adds localized masks, paired experts, quality heads, and joint routing.
# See README.md for the paper citation and LICENSE.upstream for the upstream notice.
#%%
from typing import Dict, List, Tuple
from torch import nn
import numpy as np
import torch


def mask_regularization_loss(fine_mask, fine_logits):
    continuity = (fine_mask[:, 1:] - fine_mask[:, :-1]).abs().mean()
    barrier = (torch.nn.functional.softplus(fine_logits - 6.) +
               torch.nn.functional.softplus(-fine_logits - 6.)).mean()
    return fine_mask.mean() + continuity + barrier


def quality_supervision_loss(logits, targets):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.detach())


def refine_routing_weights(base_fuse, s_vis, s_qry, s_qua, head, quality_lambda, beta):
    """Same continuous fusion for training pairs and inference candidates."""
    boost = (1. + quality_lambda * (s_qua - .5)).clamp(0., 2.)
    result = base_fuse * boost
    if head is not None:
        features = torch.stack([s_vis, s_qry, s_qua], dim=-1)
        refine = (1. + beta * torch.tanh(head(features))).clamp(0., 2.)
        result = result * refine
    return result


class Attention(nn.Module):
    def __init__(self, inp1_dim:int, inp2_dim:int, qk_dim:int, v_dim:int, head_n:int,
                 add_bias_q = True, add_bias_k = True, add_bias_v = True) -> None:
        super().__init__()
        assert qk_dim % head_n == 0
        self.head_n = head_n
        self.qk_head_dim = qk_dim // head_n
        self.v_head_dim = v_dim // head_n
        self.scale_factor = 1 / (self.qk_head_dim ** 0.5)
        self.q_mlp = nn.Linear(inp1_dim, qk_dim, add_bias_q)
        self.k_mlp = nn.Linear(inp2_dim, qk_dim, add_bias_k)
        self.v_mlp = nn.Linear(inp2_dim, v_dim, add_bias_v)
        self.reset_parameters()
    
    def forward(self, inp1:torch.Tensor, inp2:torch.Tensor, rescale_with_score = False):
        # inp1: [b, l1, d1]; inp2: [b, l2, d2];
        b, l1, _ = inp1.shape
        b, l2, _ = inp2.shape
        q = self.q_mlp(inp1).reshape(b, l1, self.head_n, self.qk_head_dim)
        k = self.k_mlp(inp2).reshape(b, l2, self.head_n, self.qk_head_dim)
        v = self.v_mlp(inp2).reshape(b, l2, self.head_n, self.v_head_dim)
        s = torch.softmax(torch.einsum('blhd,bmhd->blmh', q, k) * self.scale_factor, 2) # [b,l,m,h]
        res = torch.einsum('blmh,bmhd->blhd', s, v) # [b,l,h,d]
        if not rescale_with_score:
            return res.reshape(b, l1, -1) # [b,l,h*d]
        # rescale variance back to v's original variance
        res = res / torch.sum(s**2, 2).unsqueeze(-1)**0.5 # [b,l,h,d]
        return res.reshape(b, l1, -1)

    def reset_parameters(self):
        self.q_mlp.reset_parameters()
        self.k_mlp.reset_parameters()
        self.v_mlp.reset_parameters()

class QVExtractor(nn.Module):
    def __init__(self, eqe_n, inpt_reps_dim, module_dim, cross_att_head_n, vision_tok_n, vis_prot = False) -> None:
        super().__init__()
        # extract query for extract vision
        self.layer_norm1 = nn.LayerNorm(inpt_reps_dim)
        self.eqe1 = nn.Parameter(torch.zeros(1, eqe_n, module_dim)) # vision edit query extractor
        self.ca_query_info_ext1 = Attention(module_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        self.ca_vision_info_ext = Attention(module_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        # extract only for query
        self.layer_norm2 = nn.LayerNorm(inpt_reps_dim)
        self.eqe2 = nn.Parameter(torch.zeros(1, eqe_n, module_dim)) # vision edit query extractor
        self.ca_query_info_ext2 = Attention(module_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        # prototype vision representation
        if vis_prot:
            self.vis_rep_prot = nn.Parameter(torch.zeros(1, vision_tok_n, inpt_reps_dim)) # used for relative similarity comparison
        self.reset_parameters()

    def extract_vision(self, query_reps:torch.Tensor, vision_reps:torch.Tensor)->torch.Tensor:
        # vision_reps: [b, l1, d], query_reps: [b, l2, d]
        assert len(query_reps) == len(vision_reps) == 1
        query_reps, vision_reps = self.layer_norm1(query_reps), self.layer_norm1(vision_reps)
        eqr = self.ca_query_info_ext1(self.eqe1, query_reps) # [b, eqe_n, module_dim]
        evr = self.ca_vision_info_ext(eqr, vision_reps) # [b, eqe_n, module_dim]
        return evr # [b, eqe_n, module_dim]
    
    def extract_query(self, query_reps:torch.Tensor)->torch.Tensor:
        # query_reps: [b,l,d]
        assert len(query_reps) == 1
        query_reps = self.layer_norm2(query_reps)
        eqr = self.ca_query_info_ext2(self.eqe2, query_reps) # [b, eqe_n, module_dim]
        return eqr # [b, eqe_n, module_dim]

    def extract_from_visprot(self, query_reps:torch.Tensor):
        return self.extract_vision(query_reps, self.vis_rep_prot)

    def forward(self):
        raise
    
    def reset_parameters(self):
        self.layer_norm1.reset_parameters()
        nn.init.kaiming_normal_(self.eqe1)
        self.ca_query_info_ext1.reset_parameters()
        self.ca_vision_info_ext.reset_parameters()
        self.layer_norm2.reset_parameters()
        nn.init.kaiming_normal_(self.eqe2)
        self.ca_query_info_ext2.reset_parameters()
        if hasattr(self, 'vis_rep_prot'):
            nn.init.kaiming_normal_(self.vis_rep_prot)

class LowRankGenerator(nn.Module):
    def __init__(self, lora_dim, lora_rank, lora_scale, inpt_reps_dim, module_dim, cross_att_head_n) -> None:
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(1, lora_rank, module_dim)) # lora phi
        self.ca_lora = Attention(module_dim, inpt_reps_dim, module_dim, lora_dim, cross_att_head_n)
        self.layer_norm = nn.LayerNorm(inpt_reps_dim) 
        self.scale = 1 / (lora_scale * lora_rank ** 0.5)
        self.reset_parameters()

    def forward(self, inpt_reps:torch.Tensor):
        # vision_reps: [b, l, d]
        assert len(inpt_reps) == 1
        inpt_reps = self.layer_norm(inpt_reps)
        lora = self.ca_lora(self.phi, inpt_reps) * self.scale  # [b, lora_rank, hidden_size]
        return lora

    def reset_parameters(self):
        # nn.init.normal_(self.phi, 0, 1)
        nn.init.kaiming_normal_(self.phi)
        self.ca_lora.reset_parameters()
        self.layer_norm.reset_parameters()


class EditMaskLocalizer(nn.Module):
    """Coarse-to-fine mask localizer on vision tokens.

    Inputs:
        vision_reps: [b, lv, d]   — vision token representations
        query_reps:  [b, lq, d]   — query token representations
        ans_reps:    [b, la, d]   — answer token representations
    Outputs (dict):
        coarse_score:    [b, lv]  — raw relevance logit per vision token
        fine_mask:       [b, lv]  — spatially-smooth continuous mask in (0, 1)
        necessity_score: [b, lv]  — per-token necessity estimate in (0, 1)
        shortlist_idx:   [b, k]   — top-k indices   (when return_shortlist=True)
        shortlist_mask:  [b, lv]  — binary shortlist (when return_shortlist=True)
    """

    def __init__(self, inpt_reps_dim: int, module_dim: int, cross_att_head_n: int,
                 shortlist_ratio: float = 0.5) -> None:
        super().__init__()
        self.shortlist_ratio = shortlist_ratio
        self.ln_v = nn.LayerNorm(inpt_reps_dim)
        self.ln_q = nn.LayerNorm(inpt_reps_dim)
        self.ln_a = nn.LayerNorm(inpt_reps_dim)
        self.ca_v2q = Attention(inpt_reps_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        self.ca_v2a = Attention(inpt_reps_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        self.coarse_proj = nn.Linear(module_dim, 1)
        self.fine_conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )
        self.nec_proj = nn.Sequential(
            nn.Linear(inpt_reps_dim + module_dim, module_dim),
            nn.GELU(),
            nn.Linear(module_dim, 1),
        )
        self.reset_parameters()

    def forward(self, vision_reps: torch.Tensor, query_reps: torch.Tensor,
                ans_reps: torch.Tensor = None, shortlist_topk: int = None,
                return_shortlist: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_reps: [b, lv, d] — vision token representations
            query_reps:  [b, lq, d] — query token representations
            ans_reps:    [b, la, d] or None — answer token representations
                         When None (inference retrieval mode), only v2q branch is used.
        """
        b, lv, _ = vision_reps.shape
        v = self.ln_v(vision_reps)                                      # [b, lv, d]
        q = self.ln_q(query_reps)                                       # [b, lq, d]
        v2q = self.ca_v2q(v, q)                                         # [b, lv, module_dim]
        
        if ans_reps is not None:
            # Training / edit-time mode: use both v2q and v2a
            a = self.ln_a(ans_reps)                                     # [b, la, d]
            v2a = self.ca_v2a(v, a)                                     # [b, lv, module_dim]
            fused = v2q + v2a                                           # [b, lv, module_dim]
        else:
            # Inference retrieval mode: only v2q (v2a branch zeroed out)
            fused = v2q                                                 # [b, lv, module_dim]
        
        coarse_score = self.coarse_proj(fused).squeeze(-1)              # [b, lv]
        refined = self.fine_conv(coarse_score.unsqueeze(1)).squeeze(1)  # [b, lv]
        fine_logits = refined + coarse_score
        fine_mask = torch.sigmoid(6. * torch.nn.functional.softsign(fine_logits / 6.))               # [b, lv]
        nec_inp = torch.cat([v, fused], dim=-1)                         # [b, lv, d+module_dim]
        necessity_logits = self.nec_proj(nec_inp).squeeze(-1)
        necessity_score = torch.sigmoid(necessity_logits)                         # [b, lv]
        result = {
            'coarse_score': coarse_score,
            'fine_mask': fine_mask, 'fine_logits': fine_logits,
            'necessity_score': necessity_score, 'necessity_logits': necessity_logits,
        }
        if return_shortlist:
            k = shortlist_topk or max(1, int(lv * self.shortlist_ratio))
            _, topk_idx = coarse_score.topk(k, dim=-1)                  # [b, k]
            shortlist_mask = torch.zeros_like(fine_mask, dtype=torch.bool)
            shortlist_mask.scatter_(1, topk_idx, True)                   # [b, lv]
            result['shortlist_idx'] = topk_idx
            result['shortlist_mask'] = shortlist_mask
        return result

    def reset_parameters(self):
        self.ln_v.reset_parameters()
        self.ln_q.reset_parameters()
        self.ln_a.reset_parameters()
        self.ca_v2q.reset_parameters()
        self.ca_v2a.reset_parameters()
        nn.init.kaiming_normal_(self.coarse_proj.weight)
        nn.init.zeros_(self.coarse_proj.bias)
        for seq in (self.fine_conv, self.nec_proj):
            for m in seq:
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()


class DualLowRankGenerator(nn.Module):
    """Shared-trunk dual-head generator for support / suppression low-rank experts.

    Input:
        masked_edit_reps: [b, l, d]  — mask-weighted edit representations
    Output:
        support_c:  [b, lora_rank, lora_dim]
        support_r:  [b, lora_rank, lora_dim]
        suppress_c: [b, lora_rank, lora_dim]
        suppress_r: [b, lora_rank, lora_dim]
    """

    def __init__(self, lora_dim: int, lora_rank: int, lora_scale: float,
                 inpt_reps_dim: int, module_dim: int, cross_att_head_n: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(inpt_reps_dim)
        self.scale = 1 / (lora_scale * lora_rank ** 0.5)
        self.phi_sc = nn.Parameter(torch.zeros(1, lora_rank, module_dim))
        self.ca_sc = Attention(module_dim, inpt_reps_dim, module_dim, lora_dim, cross_att_head_n)
        self.phi_sr = nn.Parameter(torch.zeros(1, lora_rank, module_dim))
        self.ca_sr = Attention(module_dim, inpt_reps_dim, module_dim, lora_dim, cross_att_head_n)
        self.phi_pc = nn.Parameter(torch.zeros(1, lora_rank, module_dim))
        self.ca_pc = Attention(module_dim, inpt_reps_dim, module_dim, lora_dim, cross_att_head_n)
        self.phi_pr = nn.Parameter(torch.zeros(1, lora_rank, module_dim))
        self.ca_pr = Attention(module_dim, inpt_reps_dim, module_dim, lora_dim, cross_att_head_n)
        self.reset_parameters()

    def forward(self, masked_edit_reps: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # masked_edit_reps: [b, l, d]
        assert len(masked_edit_reps) == 1
        x = self.layer_norm(masked_edit_reps)                      # [b, l, d]
        support_c  = self.ca_sc(self.phi_sc, x) * self.scale       # [b, rank, lora_dim]
        support_r  = self.ca_sr(self.phi_sr, x) * self.scale       # [b, rank, lora_dim]
        suppress_c = self.ca_pc(self.phi_pc, x) * self.scale       # [b, rank, lora_dim]
        suppress_r = self.ca_pr(self.phi_pr, x) * self.scale       # [b, rank, lora_dim]
        return support_c, support_r, suppress_c, suppress_r

    def reset_parameters(self):
        self.layer_norm.reset_parameters()
        for name in ('phi_sc', 'phi_sr', 'phi_pc', 'phi_pr'):
            nn.init.kaiming_normal_(getattr(self, name))
        for name in ('ca_sc', 'ca_sr', 'ca_pc', 'ca_pr'):
            getattr(self, name).reset_parameters()


class MagnitudeHead(nn.Module):
    """Predict non-negative edit-residual magnitude from pooled edit reps.

    Input:  edit_reps  [b, l, d]
    Output: magnitude  [b]   (non-negative, via softplus)
    """

    def __init__(self, inpt_reps_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.pool_proj = nn.Linear(inpt_reps_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def forward(self, edit_reps: torch.Tensor) -> torch.Tensor:
        # edit_reps: [b, l, d]
        pooled = edit_reps.mean(dim=1)                              # [b, d]
        h = torch.relu(self.pool_proj(pooled))                      # [b, hidden_dim]
        return torch.nn.functional.softplus(
            self.out_proj(h).squeeze(-1))                           # [b]

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.pool_proj.weight)
        nn.init.zeros_(self.pool_proj.bias)
        nn.init.kaiming_normal_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)


class InfluenceHead(nn.Module):
    """Score how strongly the expert flips the target answer.

    Input:  edit_reps  [b, l, d]
    Output: score      [b]   in (0, 1)
    """

    def __init__(self, inpt_reps_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inpt_reps_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def logits(self, edit_reps: torch.Tensor) -> torch.Tensor:
        pooled = edit_reps.mean(dim=1)
        pooled = torch.nn.functional.layer_norm(pooled, (pooled.shape[-1],))
        return self.mlp(pooled).squeeze(-1)

    def forward(self, edit_reps: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(edit_reps))

    def reset_parameters(self):
        for m in self.mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()


class StabilityHead(nn.Module):
    """Score expert consistency under local perturbation / neighbouring samples.

    Input:  edit_reps  [b, l, d]
    Output: score      [b]   in (0, 1)
    """

    def __init__(self, inpt_reps_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inpt_reps_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def logits(self, edit_reps: torch.Tensor) -> torch.Tensor:
        pooled = edit_reps.mean(dim=1)
        pooled = torch.nn.functional.layer_norm(pooled, (pooled.shape[-1],))
        return self.mlp(pooled).squeeze(-1)

    def forward(self, edit_reps: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(edit_reps))

    def reset_parameters(self):
        for m in self.mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()


class JointRoutingHead(nn.Module):
    """Learnable refinement head for joint sparse routing.

    Takes per-expert routing features (e.g. [s_vis, s_qry, s_qua]) and outputs
    a scalar refinement logit. The final routing score / multiplier is derived
    from this logit outside the module.

    Design rationale:
    - The last linear layer is zero-initialized so that at init the output is 0.
      When used multiplicatively as `(1 + beta * tanh(logit))`, this gives
      multiplier=1 → identity. Therefore an untrained head does NOT disturb the
      legacy `rela_sim * abs_sim` routing, which is important for not hurting
      out-of-the-box performance.
    - When trained with a joint routing loss (contrastive on positive pairs
      and null/prototype column), the head learns a small correction that
      bridges vis / qry / quality signals in a single unified scalar.
    """

    def __init__(self, feat_dim: int = 3, hidden_dim: int = 16) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [..., feat_dim]; returns [...]  (last dim squeezed)
        return self.mlp(feat).squeeze(-1)

    def reset_parameters(self):
        for m in self.mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        # Zero-init output head so initial logit is 0 → multiplier=1 (identity).
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()


class EquivalentResidualComposer(nn.Module):
    """Compose dual (support / suppress) experts + magnitude into an
    equivalent single expert compatible with the original LiveEdit interface.

    Mode A  — compose_to_single (default, for pool storage):
        Input:
            support_c, support_r, suppress_c, suppress_r: [b, rank, dim]
            magnitude: [b]
        Output:
            moe_c: [b, rank, dim]   — storable in moe_cs_pool
            moe_r: [b, rank, dim]   — storable in moe_rs_pool

    Mode B  — compose_residual (direct residual computation):
        Input:
            inpt_reps: [1, l, dim]
            + same dual-expert inputs as Mode A
            reps_norm:  nn.Module   — shared LayerNorm from LiveEdit
            token_mask: optional [1, l] continuous mask
        Output:
            residual: [1, l, dim]
    """

    def __init__(self, lora_rank: int, module_dim: int) -> None:
        super().__init__()
        self.lora_rank = lora_rank
        self.gate_net = nn.Sequential(
            nn.Linear(4 * lora_rank, module_dim),
            nn.GELU(),
            nn.Linear(module_dim, 2 * lora_rank),
        )
        self.reset_parameters()

    def compose_to_single(self, support_c: torch.Tensor, support_r: torch.Tensor,
                          suppress_c: torch.Tensor, suppress_r: torch.Tensor,
                          magnitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # support_c / suppress_c: [b, rank, dim]; magnitude: [b]
        feat = torch.cat([support_c.mean(-1), support_r.mean(-1),
                          suppress_c.mean(-1), suppress_r.mean(-1)], -1)   # [b, 4*rank]
        w = self.gate_net(feat)                                             # [b, 2*rank]
        w_s = w[:, :self.lora_rank].unsqueeze(-1)                           # [b, rank, 1]
        w_p = w[:, self.lora_rank:].unsqueeze(-1)                           # [b, rank, 1]
        moe_c = w_s * support_c - w_p * suppress_c                         # [b, rank, dim]
        moe_r = w_s * support_r - w_p * suppress_r                         # [b, rank, dim]
        mag = magnitude.view(-1, 1, 1)                                      # [b, 1, 1]
        moe_r = moe_r * mag                                                # [b, rank, dim]
        return moe_c, moe_r

    def compose_residual(self, inpt_reps: torch.Tensor,
                         support_c: torch.Tensor, support_r: torch.Tensor,
                         suppress_c: torch.Tensor, suppress_r: torch.Tensor,
                         magnitude: torch.Tensor,
                         reps_norm: nn.Module,
                         token_mask: torch.Tensor = None) -> torch.Tensor:
        # inpt_reps: [1, l, dim]; token_mask: optional [1, l]
        assert len(inpt_reps) == 1
        moe_c, moe_r = self.compose_to_single(
            support_c, support_r, suppress_c, suppress_r, magnitude)
        x = reps_norm(inpt_reps)[0]                                        # [l, dim]
        x = torch.relu(torch.einsum('ld,mrd->lmr', x, moe_c))             # [l, m, rank]
        x = torch.einsum('lmr,mrd->ld', x, moe_r)                         # [l, dim]
        residual = x.unsqueeze(0)                                           # [1, l, dim]
        if token_mask is not None:
            residual = residual * token_mask.unsqueeze(-1)                  # [1, l, dim]
        return residual

    def forward(self, support_c: torch.Tensor, support_r: torch.Tensor,
                suppress_c: torch.Tensor, suppress_r: torch.Tensor,
                magnitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compose_to_single(support_c, support_r,
                                      suppress_c, suppress_r, magnitude)

    def reset_parameters(self):
        for m in self.gate_net:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        with torch.no_grad():
            self.gate_net[-1].weight.zero_()
            self.gate_net[-1].bias.fill_(1.0)

