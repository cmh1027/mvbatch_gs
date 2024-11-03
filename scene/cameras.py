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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from torchvision.utils import save_image
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", mask_height=0, mask_width=0):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        
        self.frequency_map = torch.fft.fft2(self.original_image[None]).squeeze().abs()

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, 
                                                     W=self.image_width, H=self.image_height).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        
    def depth_map_to_3d(self, depth_map):
        """
        Convert a depth map to 3D coordinates using intrinsic camera parameters.

        :param depth_map: 2D array of depth values (H x W)
        :param intrinsic_matrix: Camera intrinsic matrix
        :return: 3D coordinates array of shape (H, W, 3)
        """
        h, w = depth_map.shape
        assert self.image_width == w and self.image_height == h, "Depth map shape does not match camera resolution"
        fx = (self.image_width / 2) / math.tan(self.FoVx / 2)
        fy = (self.image_height / 2) / math.tan(self.FoVy / 2)
        cx = self.image_width / 2
        cy = self.image_height / 2

        # Create a grid of pixel coordinates
        u, v = torch.meshgrid(torch.arange(h, device=depth_map.device), torch.arange(w, device=depth_map.device), indexing="ij")
        
        # Compute 3D coordinates
        X = (u - cx + 0.5) * depth_map / fx
        Y = (v - cy + 0.5) * depth_map / fy
        Z = depth_map

        points_cam = torch.stack((X, Y, Z), axis=-1)
        points_world = points_cam @ torch.tensor(self.R.T, device=points_cam.device).float() + self.camera_center

        return points_world
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

