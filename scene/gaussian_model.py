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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, skewness, densify_coef
from utils.reloc_utils import compute_relocation_cuda

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize



    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.vs_clone_accum = torch.empty(0)
        self.vs_split_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.vs_clone_accum,
            self.vs_split_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        vs_clone_accum, 
        vs_split_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.vs_clone_accum = vs_clone_accum
        self.vs_split_accum = vs_split_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except:
            pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)


    @property
    def num_pts(self):
        return len(self._opacity)

    def get_effective_rank(self):
        sorted_S, _ = torch.sort(self.get_scaling, dim=-1)
        S = sorted_S ** 2 # (N, 3)
        S = S / (S.sum(dim=-1, keepdim=True) + 1e-5)
        return torch.exp(-(S * torch.log(S + 1e-5)).sum(dim=-1)), sorted_S[..., 0]

    @torch.no_grad()
    def get_prob(self):
        return self.get_opacity

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def degrade_opacity(self):
        self._opacity = self.inverse_opacity_activation(self.get_opacity - 0.01)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_scale : float = 0.1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2)*init_scale)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous())
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self, opt):
        self.percent_dense = opt.percent_dense
        self.vs_clone_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.vs_split_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': opt.position_lr * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': opt.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': opt.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opt.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': opt.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': opt.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.schedulers = {}
        self.schedulers["xyz"] = get_expon_lr_func(lr_init=opt.position_lr*self.spatial_lr_scale,
                                                   lr_final=opt.position_lr*opt.position_lr_coef*self.spatial_lr_scale,
                                                   max_steps=opt.iterations)
        self.schedulers["f_dc"] = get_expon_lr_func(lr_init=opt.feature_lr,
                                                    lr_final=opt.feature_lr*opt.feature_lr_coef,
                                                    max_steps=opt.iterations)
        self.schedulers["f_rest"] = get_expon_lr_func(lr_init=opt.feature_lr/20.0,
                                                      lr_final=opt.feature_lr*opt.feature_lr_coef/20.0,
                                                      max_steps=opt.iterations)
        self.schedulers["opacity"] = get_expon_lr_func(lr_init=opt.opacity_lr,
                                                       lr_final=opt.opacity_lr*opt.opacity_lr_coef,
                                                       max_steps=opt.iterations)
        self.schedulers["scaling"] = get_expon_lr_func(lr_init=opt.scaling_lr,
                                                       lr_final=opt.scaling_lr*opt.scaling_lr_coef,
                                                       max_steps=opt.iterations)
        self.schedulers["rotation"] = get_expon_lr_func(lr_init=opt.rotation_lr,
                                                        lr_final=opt.rotation_lr*opt.rotation_lr_coef, 
                                                        max_steps=opt.iterations)


    def update_learning_rate(self, iteration, schedule_all):
        ''' Learning rate scheduling per step '''
        lrs = {}
        for param_group in self.optimizer.param_groups:
            param = param_group["name"]
            if param != "xyz" and not schedule_all: continue
            param_group['lr'] = self.schedulers[param](iteration)
            lrs[param] = param_group['lr']
        return lrs

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous())
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda"))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.vs_clone_accum = self.vs_clone_accum[valid_points_mask]
        self.vs_split_accum = self.vs_split_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.vs_clone_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.vs_split_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 

        return optimizable_tensors

    
    def _update_params(self, idxs, ratio=None):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]


    def _sample_alives(self, probs, num, alive_indices=None, replacement=True):
        sampled_idxs = torch.multinomial(probs, num, replacement=replacement) 
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs, minlength=self._xyz.shape[0]).unsqueeze(-1) # [P, 1]
        return sampled_idxs, ratio
    

    def relocate_gs(self, opt, dead_mask=None):
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return
        
        num_gs = dead_indices.shape[0]
        probs = self.get_prob()[alive_indices, 0]
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=num_gs)
        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
        ) = self._update_params(idx, ratio=ratio)

        self._opacity[idx] = self._opacity[dead_indices]
        self._scaling[idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=idx) 
        
    def add_new_gs(self, opt, tb_writer, cap_max, add_ratio=0.05, iteration=None):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int((1+add_ratio) * current_num_points))
        num_gs_total = max(0, target_num - current_num_points)

        if num_gs_total <= 0:
            return 0
        num_gs = int(num_gs_total * add_ratio / (add_ratio))

        probs = self.get_prob().squeeze(-1) 
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        ) = self._update_params(idx, ratio=ratio)

        self._opacity[idx] = new_opacity
        self._scaling[idx] = new_scaling

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self.replace_tensors_to_optimizer(inds=idx)
    
    ########## original 3dgs ##########


    def reset_opacity(self, opacity):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def split(self, grads, opt, scene_extent, add_pts_count, N=2):
        n_init_points = self.get_xyz.shape[0]
        if add_pts_count <= 0 and opt.predictable_growth: return
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        if opt.predictable_growth:
            padded_grad[~(torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)] = 0.
            _, selected_idx = torch.topk(padded_grad.squeeze(), min(add_pts_count, len(padded_grad)))
            selected_pts_mask = torch.zeros_like(padded_grad).to(torch.bool)
            selected_pts_mask[selected_idx] = True
        else:
            available = opt.max_points - self.get_xyz.shape[0]
            if available <= 0: return
            if opt.split_original:
                threshold = opt.densify_grad_clone_threshold
            else:
                threshold = opt.densify_grad_split_threshold
            selected_pts_mask = torch.where(padded_grad >= threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
            if available < selected_pts_mask.sum():
                selected_pts_mask[selected_pts_mask.nonzero()[available:]] = False

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def clone(self, grads, opt, scene_extent, add_pts_count):
        if add_pts_count <= 0 and opt.predictable_growth: return
        # Extract points that satisfy the gradient condition
        if opt.predictable_growth:
            grads[~(torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)] = 0.
            _, selected_idx = torch.topk(grads.squeeze(), min(add_pts_count, len(grads)))
            selected_pts_mask = torch.zeros_like(grads.squeeze()).to(torch.bool)
            selected_pts_mask[selected_idx] = True
        else:
            available = opt.max_points - self.get_xyz.shape[0]
            if available <= 0: return
            threshold = opt.densify_grad_clone_threshold
            selected_pts_mask = torch.where(grads.squeeze() >= threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
            if available < selected_pts_mask.sum():
                selected_pts_mask[selected_pts_mask.nonzero()[available:]] = False

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify(self, tb_writer, opt, extent, add_pts_count, iteration):
        grads_clone = self.vs_clone_accum / self.denom
        grads_clone[grads_clone.isnan()] = 0.0
        grads_split = self.vs_split_accum / self.denom
        grads_split[grads_split.isnan()] = 0.0
        
        clone_N = len(grads_clone[torch.max(self.get_scaling, dim=1).values <= self.percent_dense*extent])
        split_N = len(grads_split[torch.max(self.get_scaling, dim=1).values > self.percent_dense*extent])

        ratio = add_pts_count / self.num_pts
        clone_add_count = int(clone_N * ratio)
        split_add_count = int(split_N * ratio)
        self.clone(grads_clone, opt, extent, clone_add_count)
        self.split(grads_split, opt, extent, split_add_count)
        torch.cuda.empty_cache()

    def prune(self, tb_writer, opt, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()


    def add_densification_stats(self, vs_clone, vs_split, update_filter, denom):
        self.vs_clone_accum[update_filter] += vs_clone[update_filter]
        self.vs_split_accum[update_filter] += vs_split[update_filter]
        self.denom[update_filter] += denom[update_filter]