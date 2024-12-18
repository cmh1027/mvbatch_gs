from typing import NamedTuple
import torch.nn as nn
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward



class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, gt, mask, normalize):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(pred, gt, mask, normalize)
        ctx.save_for_backward(pred.detach(), gt, mask, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        pred, gt, mask, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        dL_dmap = opt_grad
        grad = fusedssim_backward(pred, gt, mask, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return grad, None, None, None

def fused_ssim(pred, gt, mask, normalize=True):
    if len(pred.shape) == 3:
        pred = pred[None]
        gt = gt[None]
    B, C, H, W = pred.shape
    if mask is None:
        mask = torch.ones(B, H, W, device=torch.device('cuda'))
    else:
        if len(mask.shape) == 2:
            mask = mask[None]
    return FusedSSIMMap.apply(pred, gt, mask, normalize)
