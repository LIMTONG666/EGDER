"""Numerical regressions for the EGDER implementation."""
import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(4)
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATH=Path(__file__).resolve().parents[1]/'egder/modules.py'
spec=importlib.util.spec_from_file_location('egder_modules',PATH)
modules=importlib.util.module_from_spec(spec);spec.loader.exec_module(modules)
class EGDERTests(unittest.TestCase):
    def test_mask_recovers_from_both_saturated_tails(self):
        for sign in (1,-1):
            torch.manual_seed(2026)
            model=modules.EditMaskLocalizer(16,8,2).to(DEVICE)
            with torch.no_grad():
                model.coarse_proj.weight.zero_();model.coarse_proj.bias.fill_(80*sign)
                for p in model.fine_conv.parameters():p.zero_()
            x=torch.randn(1,4,16,device=DEVICE)
            out=model(x,x,x)
            self.assertTrue(bool(((out['fine_mask']>0)&(out['fine_mask']<1)).all()),
                            'raw sigmoid saturation destroys the visual mask')
            loss=modules.mask_regularization_loss(out['fine_mask'],out['fine_logits'])
            loss.backward()
            grad=float(model.coarse_proj.bias.grad)
            self.assertTrue(torch.isfinite(model.coarse_proj.bias.grad).all())
            self.assertGreater(sign*grad,0.,'barrier must push both tails toward finite logits')
    def test_quality_loss_learns_failures_without_sigmoid_dead_gradients(self):
        logits=torch.tensor([100.,-100.],device=DEVICE,requires_grad=True)
        targets=torch.tensor([0.,1.],device=DEVICE)
        loss=modules.quality_supervision_loss(logits,targets)
        loss.backward()
        self.assertGreater(float(logits.grad[0]),0)
        self.assertLess(float(logits.grad[1]),0)
    def test_joint_fusion_trains_the_quality_feature_used_by_inference(self):
        torch.manual_seed(2026)
        head=modules.JointRoutingHead(3,4).to(DEVICE)
        with torch.no_grad():head.mlp[-1].weight.fill_(.1)
        vis=torch.tensor([[1.,2.,3.]],device=DEVICE)
        qry=torch.tensor([[.2,.4,.1]],device=DEVICE)
        quality=torch.tensor([[.1,.5,.9]],device=DEVICE)
        base=qry.softmax(-1)*qry.sigmoid()
        result=modules.refine_routing_weights(base,vis,qry,quality,head,.1,.3)
        expected=base*(1+.1*(quality-.5))*(1+.3*head(torch.stack((vis,qry,quality),-1)).tanh())
        torch.testing.assert_close(result,expected)
        result.square().mean().backward()
        self.assertGreater(float(head.mlp[0].weight.grad[:,2].abs().max()),0.)
    def test_quality_normalizes_frozen_hidden_state_scale(self):
        torch.manual_seed(2026)
        head=modules.InfluenceHead(16,8).to(DEVICE)
        x=torch.randn(1,4,16,device=DEVICE)
        torch.testing.assert_close(head.logits(x),head.logits(x*100),rtol=1e-3,atol=1e-4)
if __name__=='__main__':unittest.main()
