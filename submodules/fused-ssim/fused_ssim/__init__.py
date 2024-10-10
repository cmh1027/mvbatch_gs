from typing import NamedTuple
import torch.nn as nn
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward

allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, mask, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer = fusedssim(C1, C2, img1, img2, mask, train)
        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]
        ctx.save_for_backward(img1.detach(), img2, mask, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, mask, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, mask, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer)
        return None, None, grad, None, None, None, None

def fused_ssim(img1, img2, mask, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    img1 = img1 * mask
    img2 = img2 * mask
    assert padding in allowed_padding
    map = FusedSSIMMap.apply(C1, C2, img1, img2, mask, padding, train)
    return map
