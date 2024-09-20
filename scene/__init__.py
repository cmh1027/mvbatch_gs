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
from utils.graphics_utils import getC2W
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
            C2W = getC2W(cam.R, cam.T)
            Rs.append(cam.R[:3, :3])
            Ts.append(C2W[:3, 3])
        Rs = torch.from_numpy(np.stack(Rs)).float()
        Ts = torch.from_numpy(np.stack(Ts)).float()
        T_distance = torch.cdist(Ts, Ts) # (N, N)
        R_distance = torch.zeros((len(Rs), len(Rs)))  # Initialize an NxN matrix to store distances # (N,N)
        for i in range(len(Rs)):
            for j in range(i+1, len(Rs)):
                # R_ij = torch.matmul(Rs[i], Rs[j])
                # trace_Rij = torch.trace(R_ij)
                # distance = torch.arccos((trace_Rij - 1) / 2) # (0, pi)
                distance = (torch.dot(Rs[i][2, ...], Rs[j][2, ...]) + 1) / 2 # [0, 1]
                R_distance[i, j] = distance
                R_distance[j, i] = distance
        
        R_max = R_distance.max(dim=1, keepdim=True)[0]
        R_distance = -torch.log((R_max - R_distance) / R_max + 1e-6) # (0, inf)
        T_max = T_distance.max(dim=1, keepdim=True)[0]
        T_distance = -torch.log((T_max - T_distance) / T_max + 1e-6)

        t_coef, r_coef = args.t_coef / (args.t_coef + args.r_coef), args.r_coef / (args.t_coef + args.r_coef)
        TR_distance = t_coef * T_distance + r_coef * R_distance
        self.TR_max_prob = TR_distance.clone()
        self.TR_max_prob.fill_diagonal_(0.)
        self.TR_max_prob = self.TR_max_prob / self.TR_max_prob.sum(dim=1, keepdim=True)

        self.TR_min_prob = 1 / TR_distance
        self.TR_min_prob.fill_diagonal_(0.)
        self.TR_min_prob = self.TR_min_prob / self.TR_min_prob.sum(dim=1, keepdim=True)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def sample_cameras(self, idx, N=2, strategy="min"):
        if strategy == "min":
            indices = self.TR_min_prob[idx].multinomial(num_samples=N-1, replacement=False)
        elif strategy == "max":
            indices = self.TR_max_prob[idx].multinomial(num_samples=N-1, replacement=False)
        elif strategy == "random":
            indices = torch.randperm(self.TR_max_prob.shape[0])
            indices = indices[indices != idx][:N-1]
        else:
            raise NotImplementedError
        selected = torch.cat([torch.tensor([idx]), indices])
        return selected
    
