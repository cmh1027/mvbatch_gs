#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from fused_ssim import fused_ssim
import math
window = None

loss_dict = {
    "l1" : lambda x1, x2: torch.abs(x1 - x2),
    "l2" : lambda x1, x2: 0.5 * (x1 - x2) ** 2
}

def pixel_loss(pred, gt, ltype="l1", weight=None):
    loss = loss_dict[ltype](pred, gt)
    if weight is not None:
        loss = loss * weight
    return loss.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, normalize=True):
    return fused_ssim(img1, img2, mask, normalize=normalize)

# def ssim(img1, img2, window_size=11, mask=None):
#     channel = img1.size(-3)
#     global window
#     if window is None:
#         window = create_window(window_size, channel)
#         if img1.is_cuda:
#             window = window.cuda()
#     return _ssim(img1, img2, window, window_size, channel, mask=mask)

def conv2d(img, window, padding, groups, mask=None):
    if mask is None:
        return F.conv2d(img, window, padding=padding, groups=groups)
    else:
        masked_window = F.conv2d(mask, window, padding=padding)
        return F.conv2d(img, window, padding=padding, groups=groups) / (masked_window + torch.finfo(torch.float32).eps)

def _ssim(img1, img2, window, window_size, channel, mask=None):
    if mask is not None:
        mask = mask[None, ...]
        img1 = img1 * mask
        img2 = img2 * mask

    mu1 = conv2d(img1, window, window_size // 2, channel, mask=mask)
    mu2 = conv2d(img2, window, window_size // 2, channel, mask=mask)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, window_size // 2, channel, mask=mask) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, window_size // 2, channel, mask=mask) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, window_size // 2, channel, mask=mask) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map