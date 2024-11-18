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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    opacities,
    scales,
    rotations,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        opacities,
        scales,
        rotations,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        opacities,
        scales,
        rotations,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.mask,
            raster_settings.low_pass,
            raster_settings.time_check,
            raster_settings.debug
        )
        
        (
            num_rendered, 
            batch_num_rendered, 
            rendered_color, 
            rendered_depth, 
            residual_trans, 
            radii, 
            cacheBuffer, 
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            mask,
            measureTime, 
            preprocessTime, 
            renderTime
        ) = _C.rasterize_gaussians(*args)

        raster_settings.log_buffer["R"] = num_rendered
        raster_settings.log_buffer["BR"] = batch_num_rendered
        raster_settings.log_buffer["forward_measureTime"] = measureTime
        raster_settings.log_buffer["forward_preprocessTime"] = preprocessTime
        raster_settings.log_buffer["forward_renderTime"] = renderTime
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.batch_num_rendered = batch_num_rendered
        ctx.save_for_backward(means3D, scales, rotations, radii, sh, cacheBuffer, geomBuffer, binningBuffer, imgBuffer, mask)
        radii = radii.max(dim=0).values # (B, N) => (N,)
        return rendered_color, radii, rendered_depth, residual_trans

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth, grad_trans):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        batch_num_rendered = ctx.batch_num_rendered
        raster_settings = ctx.raster_settings
        means3D, scales, rotations, radii, sh, cacheBuffer, geomBuffer, binningBuffer, imgBuffer, mask = ctx.saved_tensors
        
        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_depth,
                grad_trans,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                cacheBuffer,
                geomBuffer,
                num_rendered,
                batch_num_rendered,
                binningBuffer,
                imgBuffer,
                mask,
                raster_settings.low_pass,
                raster_settings.grad_sep,
                raster_settings.time_check,
                raster_settings.debug)
        
        (
            grad_means2D, 
            grad_opacities, 
            grad_means3D, 
            grad_sh, 
            grad_scales, 
            grad_rotations, 
            denom,
            preprocessTime,
            renderTime
        ) = _C.rasterize_gaussians_backward(*args)
        raster_settings.log_buffer["denom"] = denom
        raster_settings.log_buffer["backward_preprocessTime"] = preprocessTime
        raster_settings.log_buffer["backward_renderTime"] = renderTime
        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_opacities,
            grad_scales,
            grad_rotations,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : torch.Tensor
    tanfovy : torch.Tensor
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    debug : bool
    mask : torch.Tensor
    low_pass : float
    log_buffer: dict
    normalize_grad2D : bool
    grad_sep: bool
    time_check: bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, shs, opacities, scales, rotations):
        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            opacities,
            scales, 
            rotations,
            raster_settings, 
        )

def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale 

def make_category_mask(mask, H, W, B):
    return _C.make_category_mask(mask, H, W, B)