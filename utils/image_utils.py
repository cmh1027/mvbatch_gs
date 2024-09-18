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
import matplotlib
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def apply_float_colormap(image, colormap):
    if colormap == "default":
        colormap = "turbo"
    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[image_long[..., 0]]

def apply_colormap(image):
    # default for rgb images
    if image.shape[-1] == 3:
        return image
    # rendering depth outputs
    if image.shape[-1] == 1 and torch.is_floating_point(image):
        output = image
        output = torch.clip(output, 0, 1)
        return apply_float_colormap(output, "default")

    raise NotImplementedError

def rescale_gt_depth(pred, gt):
    pred_m, pred_M = pred.min(), pred.max()
    gt_m, gt_M = gt.min(), gt.max()
    gt = (gt - gt_m) / (gt_M - gt_m) * (pred_M - pred_m) + pred_m
    return gt

def apply_depth_colormap(depth, near_plane=None, far_plane=None):
    near_plane = near_plane if near_plane is not None else float(torch.min(depth))
    far_plane = far_plane if far_plane is not None else float(torch.max(depth))
    depth = torch.clip(depth, min=near_plane, max=far_plane)
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    colored_image = apply_colormap(depth)
    return colored_image.permute(2, 0, 1)
