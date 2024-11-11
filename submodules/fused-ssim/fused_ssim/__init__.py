from typing import NamedTuple
import torch.nn as nn
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward



class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2, mask):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer = fusedssim(img1, img2, mask)
        ctx.save_for_backward(img1.detach(), img2, mask, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer)

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, mask, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer = ctx.saved_tensors
        dL_dmap = opt_grad
        grad = fusedssim_backward(img1, img2, mask, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer)
        return grad, None, None

def fused_ssim(img1, img2, mask):
    _, C, H, W = img1.shape
    if mask is None:
        mask = torch.ones(1, H, W, device=torch.device('cuda'))
    return FusedSSIMMap.apply(img1, img2, mask)
