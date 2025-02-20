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
import torch
from torch_linear_assignment import batch_linear_assignment

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], skip_train=False, skip_test=False, sampling_mode='None'):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.sampling_mode = sampling_mode

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
            

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class ViewpointSampler:
    def __init__(self, gaussians : GaussianModel, num_viewpoints, batch_size, viewpoint_sampling):
        self.gaussians = gaussians
        self.num_viewpoint = num_viewpoints
        self.batch_size = batch_size
        self.viewpoint_sampling = viewpoint_sampling
        self.refill_queue(init=True)
        self.viewpoint_sample_count = torch.ones(num_viewpoints, dtype=torch.int32, device='cuda')

    def sample_cameras(self):
        cam_idxs = self.queue[0]
        self.queue = self.queue[1:]
        if len(self.queue) == 0:
            self.refill_queue()
        self.viewpoint_sample_count[cam_idxs] += 1
        return cam_idxs

    def refill_queue(self, init=False):
        if self.batch_size == 1:
            self.queue = torch.randperm(self.num_viewpoint, device='cuda').reshape(-1, 1) 
        else:
            if init or not self.viewpoint_sampling:
                self.queue = torch.randperm(self.num_viewpoint, device='cuda')
                if len(self.queue) % self.batch_size != 0:
                    pad_size = self.batch_size - len(self.queue) % self.batch_size
                    pad_values = self.queue[:pad_size]
                    self.queue = torch.cat([self.queue, pad_values], dim=0)
                self.queue = self.queue.reshape(-1, self.batch_size)
            else:
                self.queue = torch.arange(self.num_viewpoint, device='cuda').reshape(-1, 1)
                feature = visibility = self.gaussians.gaussian_visibility
                while self.queue.shape[1] < self.batch_size:
                    weight = self.dissim_matrix(feature) # (N, N)
                    assignment = batch_linear_assignment(-weight[None])[0] # (N,)
                    self.queue = torch.cat([
                        self.queue,
                        self.queue[assignment]
                    ], dim=-1)
                    feature = torch.stack([visibility[k] for k in self.queue.unbind(dim=-1)]).max(dim=0).values
                self.queue = self.queue[torch.randperm(self.queue.shape[0], device='cuda')]
                self.gaussians.reset_voxel()
                self.gaussians.reset_gaussian_visibility(self.num_viewpoint)
                

    def dissim_matrix(self, feature): # (N, P)
        feature = feature.float()
        difference = feature @ (1-feature).permute(1, 0) # (N, N)
        normalize_factor = feature.sum(dim=-1)[None, ...] + feature.sum(dim=-1)[..., None] - difference # union
        return  difference / normalize_factor