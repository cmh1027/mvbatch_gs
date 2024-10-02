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
import json
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, mask_schedule
from tqdm import tqdm
from utils.image_utils import psnr, apply_depth_colormap, psnr_freq
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
import random
from torchvision.utils import save_image
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from render import render_sets
from metrics import evaluate
from datetime import timedelta
import time


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, forced_exit):
    if opt.gs_type == "original":
        dataset.init_scale = 1
        if opt.max_points == -1:
            opt.max_points = dataset.cap_max
    
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    opt.batch_size = min(opt.batch_size, len(scene.getTrainCameras()))
    print(f"BATCH SIZE : {opt.batch_size}")

    dummy_cam = scene.getTrainCameras()[0]
    H, W = dummy_cam.image_height, dummy_cam.image_width
    if opt.batch_size == 1 and opt.single_partial_rays:
        n_rays = (H * W) // opt.single_partial_rays_denom
    else:
        if opt.batch_partition:
            n_rays = (H * W) // opt.batch_size
        else:
            n_rays = (H * W) 

    if opt.mask_grid:
        partial_n_rays = n_rays // (opt.mask_width * opt.mask_height)
        partial_height = (H + opt.mask_height - 1) // opt.mask_height
        partial_width = (W + opt.mask_width - 1) // opt.mask_width
        if opt.priority_mask_sampling:
            loss_accum_list = torch.stack([torch.zeros(partial_height, partial_width, device=torch.device('cuda')) for _ in range(len(scene.getTrainCameras()))])
            loss_accum_list.fill_(1e+8)
            
    print(f"Image ({H} x {W} = {H * W}), n_rays : {n_rays}")
    if opt.mask_grid:
        print(f"tile size {opt.mask_height} x {opt.mask_width}")

    start_time = time.time()

    for iteration in range(first_iter, opt.iterations + 1): 
        gt_images = []
        if iteration == forced_exit:
            print("FORCED EXIT")
            break   
        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if iteration == opt.batch_until and opt.batch_size > 1:
            print("BATCH IS TURNED OFF")
            opt.batch_size = 1

        # Pick a random Camera
        viewpoints = scene.getTrainCameras().copy()
        if not viewpoint_stack:
            viewpoint_stack = list(range(len(scene.getTrainCameras())))

        cam_idx = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if opt.batch_size == 1:
            cam_idxs = [cam_idx]
        else:
            cam_idxs = scene.sample_cameras(cam_idx, N=opt.batch_size, strategy=opt.batch_sample_strategy)

        cams = [viewpoints[min(idx, len(viewpoints)-1)] for idx in cam_idxs]
        vis_ratios = []
        use_preprocess_mask = mask_schedule(opt, iteration)

        if opt.gs_type == "original":
            batch_vs = []
            batch_radii = torch.zeros_like(gaussians.get_opacity[..., 0], dtype=torch.int32)
            visibility_count = torch.zeros_like(gaussians.get_opacity[..., 0], dtype=torch.uint8)
        
        if opt.exclusive_update:
            gradient_mask = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device=torch.device('cuda'))
        else:
            gradient_mask = None



        for idx, viewpoint_cam in enumerate(cams):
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            if n_rays == (H * W):
                pmask = None
            else:
                if opt.mask_grid:
                    pmask = torch.zeros(partial_height*partial_width, dtype=torch.int32, device=torch.device('cuda'))
                    if not opt.priority_mask_sampling:
                        indices = torch.randperm(len(pmask), device=torch.device('cuda'))[:partial_n_rays]
                        pmask[indices] = 1
                    else:
                        loss_accum = loss_accum_list[idx].view(-1)
                        loss_accum = loss_accum / loss_accum.sum()
                        priority_indices = loss_accum.multinomial(num_samples=int(partial_n_rays * opt.priority_mask_ratio), replacement=False)
                        pmask[priority_indices] = 1
                        if opt.priority_mask_ratio < 1.0:
                            indices = torch.randperm(len(pmask), device=torch.device('cuda'))[:int(partial_n_rays * (1-opt.priority_mask_ratio))]
                            pmask[indices] = 1
                    pmask = pmask.view(partial_height, partial_width)
                else:
                    pmask = torch.zeros(H*W, dtype=torch.int32, device=torch.device('cuda'))
                    pmask[torch.randperm(len(pmask), device=torch.device('cuda'))[:n_rays]] = 1

            kwargs = {
                'mask' : pmask,
                'aligned_mask' : opt.mask_grid,
                'use_preprocess_mask' : use_preprocess_mask,
                'gradient_mask' : gradient_mask
            }
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, **kwargs)
            (image, depth, viewspace_point_tensor, visibility_filter, radii) = (
                render_pkg["render"][:3, ...], 
                render_pkg["depth"],
                render_pkg["viewspace_points"], 
                render_pkg["visibility_filter"], 
                render_pkg["radii"]
            )
            if opt.gs_type == "original":
                visibility_count = visibility_count + visibility_filter.to(visibility_count.dtype)
                batch_vs += [viewspace_point_tensor]
                batch_radii = torch.maximum(radii, batch_radii)

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            gt_images += [gt_image.cpu()]
            if pmask is None:
                loss_mask = None
            else:
                if opt.mask_grid:
                    loss_mask = torch.zeros(H, W, device=torch.device('cuda'))
                    pmask_expand = torch.kron(pmask, torch.ones(opt.mask_height, opt.mask_width, device=torch.device('cuda')))
                    loss_mask[:, :] = pmask_expand[:H, :W]
                else:
                    loss_mask = pmask.view(H, W).to(torch.float)

            E_u, Ll1 = l1_loss(image, gt_image, mask=loss_mask)
            if opt.mask_grid and opt.priority_mask_sampling:
                with torch.no_grad():
                    padded_E_u = torch.zeros(E_u.shape[0], partial_height * opt.mask_height, partial_width * opt.mask_width).cuda()
                    padded_E_u[:, :E_u.shape[1], :E_u.shape[2]] = E_u
                    window = torch.ones(1,E_u.shape[0],opt.mask_height,opt.mask_width).cuda()
                    loss_grid = F.conv2d(padded_E_u.unsqueeze(0), window, stride=(opt.mask_height, opt.mask_width)).squeeze()
                    if iteration % 200 == 0:
                        tb_writer.add_histogram("train/grid_loss", loss_grid.view(-1), iteration)
                    if opt.priority_mask_mode == "max":
                        loss_accum_list[idx] = torch.minimum(loss_accum_list[idx], loss_grid)
                    elif opt.priority_mask_mode == "min":
                        loss_accum_list[idx] = torch.minimum(loss_accum_list[idx], 1 / (loss_grid + 1e-6))
                    
            lambda_dssim = opt.lambda_dssim
            loss = (1.0 - lambda_dssim) * Ll1
            if lambda_dssim > 0:
                ssim_map = ssim(image, gt_image, mask=loss_mask)
                ssim_loss = 1 - ssim_map
                loss += lambda_dssim * ssim_loss.mean()
            if not opt.grad_sum:
                loss = loss / len(cams)
            loss.backward()
            
            if opt.exclusive_update:
                gradient_mask = gradient_mask & (gaussians._opacity.grad.squeeze().abs() == 0.)

        if not opt.evaluate_time:
            vis_ratios.append(((gaussians._opacity.grad != 0).sum() / len(gaussians._opacity)).item())
            if iteration % 100 == 0:
                visible_ratio = sum(vis_ratios) / len(vis_ratios)
                tb_writer.add_scalar(f'train/vis_ratio', visible_ratio, iteration)
                vis_ratios = []
                tb_writer.add_scalar(f'train/num_points', len(gaussians._xyz), iteration)

        if opt.gs_type == "mcmc":
            reg_loss = 0
            reg_loss += args.opacity_reg * torch.abs(gaussians.get_opacity).mean() 
            reg_loss += args.scale_reg * torch.abs(gaussians.get_scaling).mean() 
            reg_loss.backward()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {
                    "Loss": f"{ema_loss_for_log:.{4}f}",
                    "num_pts": len(gaussians._xyz)
                }
                if "lambda_dssim" in locals():
                    postfix["ssim_l"] = lambda_dssim
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            # Log and save
            if not opt.evaluate_time:
                training_report(opt, tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations and not opt.evaluate_time):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                if opt.gs_type == "original":
                    mask = visibility_count > 0
                    gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], batch_radii[mask])
                    batch_vs = torch.stack([t.grad for t in batch_vs], dim=0)  # (B, N, 4)
                    
                    vs = batch_vs[..., 0:2].norm(dim=-1, keepdim=True).sum(dim=0) 
                    vs_abs = batch_vs[..., 2:4].norm(dim=-1, keepdim=True).sum(dim=0) 

                    gaussians.add_densification_stats(vs, vs_abs, mask)
                    
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt, opt.opacity_reset_threshold / 2, scene.cameras_extent, size_threshold, iteration)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity(opt.opacity_reset_threshold)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if opt.gs_type == "original":
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt, opt.opacity_reset_threshold / 2, scene.cameras_extent, size_threshold, iteration)
                    elif opt.gs_type == "mcmc":
                        dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                        gaussians.relocate_gs(dead_mask=dead_mask)
                        gaussians.add_new_gs(opt, cap_max=args.cap_max, add_ratio=opt.add_ratio, iteration=iteration)

            if iteration % args.vis_iteration_interval == 0 and not opt.evaluate_time:
                os.makedirs(os.path.join(dataset.model_path, "vis"), exist_ok=True)
                with torch.no_grad():
                    render_pkg = render(viewpoints[cam_idx], gaussians, pipe, bg)
                    image = render_pkg["render"][:3, ...].cpu()
                    depth = render_pkg["depth"].cpu()
                    gt_image = viewpoints[cam_idx].original_image.cpu()
                    gt_depth = torch.zeros_like(depth)
                    pred = torch.cat([image, apply_depth_colormap(depth.permute(1, 2, 0))], dim=-1)
                    gt = torch.cat([gt_image, apply_depth_colormap(gt_depth.permute(1, 2, 0))], dim=-1)
                    figs = [pred, gt]
                    save_image(torch.cat(figs, dim=1), os.path.join(dataset.model_path, f"vis/iter_{iteration}.png"))

            if opt.log_batch and iteration % opt.log_batch_interval == 0 and not opt.evaluate_time:
                os.makedirs(os.path.join(dataset.model_path, "batch"), exist_ok=True)
                with torch.no_grad():
                    save_image(torch.cat(gt_images, dim=1), os.path.join(dataset.model_path, f"batch/{'%05d' % iteration}.png"))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if opt.gs_type == "mcmc":
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    
                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations) and not opt.evaluate_time:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if opt.batch_size == 1 and opt.log_single_partial and not opt.evaluate_time:
            if iteration % opt.log_single_partial_interval == 0:
                if opt.single_partial_rays_denom == 1:
                    image = render(viewpoint_cam, gaussians, pipe, bg)["render"][:3, :]
                    image.sum().backward()
                    visible_count_base = (gaussians._opacity.grad != 0).sum()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    log_dict = {}
                    for i in [2, 4, 8, 16, 32, 64, 128]:
                        mask_temp = torch.zeros(H*W, dtype=torch.int32, device=torch.device('cuda'))
                        mask_temp[torch.randperm(len(mask_temp), device=torch.device('cuda'))[:n_rays // i]] = 1
                        image = render(viewpoint_cam, gaussians, pipe, bg, mask=mask_temp)["render"][:3, :]
                        image.sum().backward()
                        visible_count = (gaussians._opacity.grad != 0).sum()
                        gaussians.optimizer.zero_grad(set_to_none = True)
                        log_dict[str(i)] = (visible_count / visible_count_base).item()
                    tb_writer.add_scalars(f'test/vis_ratio_over_1', log_dict, iteration)               

    end_time = time.time()
    if forced_exit is None:
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)
    if opt.evaluate_time:
        elasped_sec = end_time - start_time
        with open(os.path.join(dataset.model_path, "elapsed.txt"), "w") as f:
            f.write(str(timedelta(seconds=elasped_sec)))
    with open(os.path.join(dataset.model_path, "configs.txt"), "w") as f:
        for key, value in dataset.attr_save.items():
            f.write(f"{key} : {value}\n")
        for key, value in opt.attr_save.items():
            f.write(f"{key} : {value}\n")
        for key, value in pipe.attr_save.items():
            f.write(f"{key} : {value}\n")

def prepare_output_and_logger(args):    
    assert(args.model_path)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(opt, tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    # Report test and samples of training set
    with torch.no_grad():
        if iteration in testing_iterations:
            cameras = scene.getTestCameras()
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            psnr_freq_test = 0.0
            psnr_low_freq_test = 0.0
            psnr_high_freq_test = 0.0
            for idx, viewpoint in enumerate(cameras):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 5):
                    tb_writer.add_images('test' + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images('test' + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                l1_test += l1_loss(image, gt_image)[1].double()


                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                psnr_freq_, psnr_freq_low_, psnr_freq_high_ = psnr_freq(image, gt_image)
                psnr_freq_test += psnr_freq_
                psnr_low_freq_test += psnr_freq_low_
                psnr_high_freq_test += psnr_freq_high_

            psnr_test /= len(cameras)
            ssim_test /= len(cameras)
            l1_test /= len(cameras)
            psnr_freq_test /= len(cameras)
            psnr_low_freq_test /= len(cameras)
            psnr_high_freq_test /= len(cameras)

            if not opt.turn_off_print:
                print(f"\n[ITER {iteration}] Evaluating test: L1 {'%.5f' % l1_test} PSNR {'%.4f' % psnr_test} SSIM {'%.5f' % ssim_test}")
            if tb_writer:
                tb_writer.add_scalar('test' + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar('test' + '/loss_viewpoint - psnr', psnr_test, iteration)
                tb_writer.add_scalar('test' + '/loss_viewpoint - ssim', ssim_test, iteration)
                tb_writer.add_scalar('test' + '/loss_viewpoint - psnr_freq', psnr_freq_test, iteration)
                tb_writer.add_scalar('test' + '/loss_viewpoint - psnr_freq_low', psnr_low_freq_test, iteration)
                tb_writer.add_scalar('test' + '/loss_viewpoint - psnr_freq_high', psnr_high_freq_test, iteration)
                torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(3000, 30001, 3000)))
    parser.add_argument("--test_iteration_interval", type=int)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000,7000,30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vis_iteration_interval", "-vi", type=int, default=500)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--forced_exit", type=int)
    parser.add_argument("--render_iter", type=int)
    args = parser.parse_args(sys.argv[1:])
    if args.test_iteration_interval is not None:
        args.test_iterations = list(range(args.test_iteration_interval, 30001, args.test_iteration_interval))
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)
    
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.forced_exit)
    # All done
    
    # print("\nTraining complete.")
    render_iter = op.iterations if args.render_iter is None else args.render_iter
    if args.forced_exit is None:
        render_sets(lp.extract(args), render_iter, pp.extract(args), True, False)
        evaluate([args.model_path])