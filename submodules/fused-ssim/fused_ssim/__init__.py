from typing import NamedTuple
import torch.nn as nn
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward



class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, mask, ssim_buffer, mask_size):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer = fusedssim(C1, C2, mask_size, img1, img2, mask, ssim_buffer, True)
        ctx.save_for_backward(img1.detach(), img2, mask, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer)
        ctx.C1 = C1
        ctx.C2 = C2

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, mask, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        dL_dmap = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, mask, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer)
        return None, None, grad, None, None, None, None

def fused_ssim(img1, img2, mask, mask_size=1, ssim_buffer=None):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    _, H, W = img1.shape
    if mask is None:
        mask = torch.ones(1, H, W, device=torch.device('cuda'))
    img1 = img1 * mask
    img2 = img2 * mask
    if ssim_buffer is None:
        ssim_buffer = torch.empty(0)
    return FusedSSIMMap.apply(C1, C2, img1, img2, mask, ssim_buffer, mask_size)
