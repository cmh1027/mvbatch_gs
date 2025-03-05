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
import math
window = None

loss_dict = {
    "l1" : lambda x1, x2: torch.abs(x1 - x2),
    "l2" : lambda x1, x2: (x1 - x2) ** 2
}

def pixel_loss(pred, gt, ltype="l1"):
    loss = loss_dict[ltype](pred, gt).view(3, -1)
    return loss.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def surface_ssim(img1, img2, xyz=None, sigma=None, window_size=11):
    if xyz is not None and (sigma > 0).any():
        with torch.no_grad():
            pad_size = window_size//2
            xyz = xyz.permute(2,0,1)
            xyz_padded = torch.nn.functional.pad(xyz, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            xyz_unfolded = xyz_padded.unfold(1, window_size, 1).unfold(2, window_size, 1)
            dist_unfolded = (xyz_unfolded - xyz[:, :, :, None, None]).norm(dim=0)
            gaussian_unfolded = torch.exp(-dist_unfolded**2 / (2*sigma[:,:,None,None]**2+1e-5))
            gaussian_unfolded = gaussian_unfolded / gaussian_unfolded.sum(dim=(-1,-2), keepdim=True)
        
        img1_padded = torch.nn.functional.pad(img1, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        img1_unfolded = img1_padded.unfold(1, window_size, 1).unfold(2, window_size, 1) 
        img2_padded = torch.nn.functional.pad(img2, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        img2_unfolded = img2_padded.unfold(1, window_size, 1).unfold(2, window_size, 1) 
        
        g_img1 = gaussian_unfolded * img1_unfolded
        g_img2 = gaussian_unfolded * img2_unfolded
        mu1 = (g_img1).sum(dim=(-1,-2))
        mu2 = (g_img2).sum(dim=(-1,-2))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (g_img1 * img1_unfolded).sum(dim=(-1,-2)) - mu1_sq
        sigma2_sq = (g_img2 * img2_unfolded).sum(dim=(-1,-2)) - mu2_sq
        sigma12 = (g_img1 * img2_unfolded).sum(dim=(-1,-2)) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map

def ssim(img1, img2, window_size=11, mask=None):
    channel = img1.size(-3)
    global window
    if window is None:
        window = create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda()
    return _ssim(img1, img2, window, window_size, channel, mask=mask)

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