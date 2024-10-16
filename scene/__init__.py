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
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import getWorld2View2
import torch
import numpy as np
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
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

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

        Rs, Ts = [], []
        for cam in scene_info.train_cameras:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            Rs.append(cam.R[:3, :3])
            Ts.append(C2W[:3, 3])
        Rs = torch.from_numpy(np.stack(Rs)).float()
        Ts = torch.from_numpy(np.stack(Ts)).float()
        T_distance = torch.cdist(Ts, Ts) # (N, N)
        R_distance = torch.zeros((len(Rs), len(Rs)))  # Initialize an NxN matrix to store distances # (N,N)
        for i in range(len(Rs)):
            for j in range(i+1, len(Rs)):
                distance = (-torch.dot(Rs[i][..., 2], Rs[j][..., 2]) + 1) / 2 # [0, 1]
                distance = (-torch.dot(Rs[i][..., 2], Rs[j][..., 2]) + 1) / 2 # [0, 1]
                R_distance[i, j] = distance
                R_distance[j, i] = distance
        # 0 => 0
        # MAX => inf
        R_max = R_distance.max(dim=1, keepdim=True)[0]
        R_distance = -torch.log((R_max - R_distance) / R_max + 1e-6) # (0, inf)
        T_max = T_distance.max(dim=1, keepdim=True)[0]
        T_distance = -torch.log((T_max - T_distance) / T_max + 1e-6)

        t_coef, r_coef = args.t_coef / (args.t_coef + args.r_coef), args.r_coef / (args.t_coef + args.r_coef)
        TR_distance = t_coef * T_distance + r_coef * R_distance
        self.TR_max_prob = TR_distance.clone().cuda()
        self.TR_max_prob.fill_diagonal_(0.)
        self.TR_max_prob = self.TR_max_prob / self.TR_max_prob.sum(dim=1, keepdim=True)

        self.TR_min_prob = (1 / TR_distance).cuda()
        self.TR_min_prob.fill_diagonal_(0.)
        self.TR_min_prob = self.TR_min_prob / self.TR_min_prob.sum(dim=1, keepdim=True)

        self.sampled_count = torch.ones(len(scene_info.train_cameras), device=torch.device('cuda'))

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def sample_cameras(self, idx, N=2, strategy="min"):
        if strategy == "min":
            indices = self.TR_min_prob[idx].multinomial(num_samples=N-1)
        elif strategy == "max":
            indices = []
            distance_vector = self.TR_max_prob[idx].clone()
            for _ in range(N-1):
                index = distance_vector.multinomial(num_samples=1).squeeze()
                indices += [index.item()]
                distance_vector = torch.minimum(distance_vector, self.TR_max_prob[index])
                distance_vector = distance_vector / distance_vector.sum()
            indices = torch.tensor(indices, device=torch.device('cuda'))
        elif strategy == "random":
            # sampled_count = self.sampled_count.clone()
            # sampled_count[idx] = 0.
            # sampled_count = sampled_count / sampled_count.sum()
            # indices = sampled_count.multinomial(num_samples=N-1)
            # self.sampled_count[indices] += 1
            indices = torch.randperm(self.TR_max_prob.shape[0], device=torch.device('cuda'))
            indices = indices[indices != idx][:N-1]
        else:
            raise NotImplementedError
        selected = torch.cat([torch.tensor([idx], device=torch.device('cuda')), indices])
        return selected
    
