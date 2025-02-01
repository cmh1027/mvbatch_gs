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

import os
import random
import json
import torch
import numpy as np

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import getWorld2View2
from utils.general_utils import gmm_kl

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], skip_train=False, skip_test=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  
            random.shuffle(scene_info.test_cameras)  

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            if not skip_train:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            if not skip_test:
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, init_scale=args.init_scale)
        
        self.viewpoint_feature = None

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def sample_cameras(self, idx, N=2, farthest=False, mode="simple"):
        if not farthest or self.viewpoint_feature is None: # random
            indices = torch.randperm(len(self.getTrainCameras()), device='cuda')
            indices = indices[indices != idx][:N-1]
            selected = torch.cat([torch.tensor([idx], device='cuda'), indices])
        else:
            """
            viewpoint_feature : (B, HS)
            """
            cam_idxs = torch.tensor([idx], device='cuda')
            masked_feature = self.viewpoint_feature & ~self.viewpoint_feature[[idx]]
            if mode == "simple":
                weight = masked_feature.sum(dim=-1).float()
                sampled = torch.multinomial(weight, N-1)
                cam_idxs = torch.cat([cam_idxs, sampled])
            elif mode == "all":
                while len(cam_idxs) < N:
                    weight = masked_feature.sum(dim=-1).float()
                    if weight.max() == 0:
                        indices = torch.randperm(len(self.getTrainCameras()), device=torch.device('cuda'))
                        sampled = indices[~torch.isin(indices, cam_idxs)][:N-len(cam_idxs)]
                        cam_idxs = torch.cat([cam_idxs, sampled])
                    else:
                        sampled = torch.multinomial(weight, 1)
                        cam_idxs = torch.cat([cam_idxs, sampled])
                        masked_feature = masked_feature & ~self.viewpoint_feature[[sampled]]
            else:
                raise NotImplementedError
            selected = cam_idxs
        return selected

    def getAllProjMatrix(self, scale=1.0):
        return torch.stack([cam.full_proj_transform for cam in self.train_cameras[scale]])

    def getAllViewMatrix(self, scale=1.0):
        return torch.stack([cam.world_view_transform for cam in self.train_cameras[scale]])
