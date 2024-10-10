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

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, 
                                                     W=self.image_width, H=self.image_height).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.translated_projs = {(0, 0):self.projection_matrix}
        for y in range(mask_height):
            for x in range(mask_width):
                if x == 0 and y == 0: continue
                self.translated_projs[(y, x)] = self._translate_proj(self, y, x)
                
    
    def translate_proj(self, offset_y, offset_x):
        return self.translated_projs[(offset_y, offset_x)]

    def _translate_proj(self, offset_y, offset_x):
        if offset_x == 0 and offset_y == 0:
            return self.full_proj_transform
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, 
                                                W=self.image_width, H=self.image_height, x_offset_pix=-offset_x, y_offset_pix=-offset_y).transpose(0,1).cuda()
        full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        return full_proj_transform
    
    def translated_gt_image(self, rendered, offset_y, offset_x):
        if offset_x == 0 and offset_y == 0:
            return self.original_image, torch.ones_like(self.original_image[0:1])
        gt_image = rendered.clone()
        unmasked = torch.zeros_like(self.original_image[0:1])
        _, H, W = gt_image.shape
        if offset_y >= 0 and offset_x >= 0:
            gt_image[:, :H-offset_y, :W-offset_x] = self.original_image[:, offset_y:, offset_x:]
            unmasked[:, :H-offset_y, :W-offset_x] = 1
        elif offset_y < 0 and offset_x >= 0:
            gt_image[:, -offset_y:, :W-offset_x] = self.original_image[:, :offset_y, offset_x:]
            unmasked[:, -offset_y:, :W-offset_x] = 1
        elif offset_y >= 0 and offset_x < 0:
            gt_image[:, :H-offset_y, -offset_x:] = self.original_image[:, offset_y:, :offset_x]
            unmasked[:, :H-offset_y, -offset_x:] = 1
        else:
            gt_image[:, -offset_y:, -offset_x:] = self.original_image[:, :offset_y, :offset_x]
            unmasked[:, -offset_y:, -offset_x:] = 1
        return gt_image, unmasked
    
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

