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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
FOV_WARN = False
def render(viewpoint_cameras, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, mask=None, normalize_grad2D=False, low_pass=0.3, separate_batch=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 2), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    if type(viewpoint_cameras) is not list:
        viewpoint_cameras = [viewpoint_cameras]

    # Set up rasterization configuration
    image_height = int(viewpoint_cameras[0].image_height)
    image_width = int(viewpoint_cameras[0].image_width)
    viewmatrix = torch.stack([cam.world_view_transform for cam in viewpoint_cameras])
    projmatrix = torch.stack([cam.full_proj_transform for cam in viewpoint_cameras])
    campos = torch.stack([cam.camera_center for cam in viewpoint_cameras])
    tanfovx = torch.tensor([math.tan(cam.FoVx * 0.5) for cam in viewpoint_cameras]).to(viewmatrix.device)
    tanfovy = torch.tensor([math.tan(cam.FoVy * 0.5) for cam in viewpoint_cameras]).to(viewmatrix.device)

    if mask is None: 
        mask = torch.empty(0, dtype=torch.int32)
    log_buffer = {}
    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        mask=mask,
        debug=pipe.debug,
        log_buffer=log_buffer,
        normalize_grad2D=normalize_grad2D,
        low_pass=low_pass,
        separate_batch=separate_batch
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    betas = pc.get_beta


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_beta, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        betas = betas)
    return_dict =  {"render": rendered_image,
                    "beta": rendered_beta,
                    "viewspace_points": screenspace_points,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "depth": depth,
                    "log_buffer": log_buffer}
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict
