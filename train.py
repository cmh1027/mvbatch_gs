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
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, rescale_gt_depth, apply_depth_colormap
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
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
    pmask = torch.zeros(dummy_cam.image_height*dummy_cam.image_width, dtype=torch.int32, device=torch.device('cuda'))
    if opt.batch_size == 1 and opt.single_partial_rays:
        n_rays = (dummy_cam.image_height * dummy_cam.image_width) // opt.single_partial_rays_denom
    else:
        if opt.batch_partition:
            n_rays = (dummy_cam.image_height * dummy_cam.image_width) // opt.batch_size
        else:
            n_rays = (dummy_cam.image_height * dummy_cam.image_width) 
    print(f"Image ({dummy_cam.image_height} x {dummy_cam.image_width} = {dummy_cam.image_height * dummy_cam.image_width}), n_rays : {n_rays}")

    for iteration in range(first_iter, opt.iterations + 1): 
        if iteration == opt.forced_exit:
            print("FORCED EXIT")
            exit(0)       
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

        cams = [viewpoints[idx] for idx in cam_idxs]
        aux_densify_grad = None
        aux_densify = opt.aux_densify and iteration >= opt.aux_densify_from_iter

        for idx, viewpoint_cam in enumerate(cams):
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            if n_rays == (dummy_cam.image_height * dummy_cam.image_width):
                pmask = None
            else:
                pmask[:] = 0
                pmask[torch.randperm(len(pmask), device=torch.device('cuda'))[:n_rays]] = 1

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, mask=pmask, aux_densify=aux_densify)
            image = render_pkg["render"][:3, :]
            depth = render_pkg["depth"]
            visibility_filter = render_pkg["visibility_filter"]

            if opt.batch_size == 1 and opt.log_single_partial and iteration % opt.log_single_partial_interval == 0:
                tb_writer.add_scalar('vis_ratio', visibility_filter.sum() / len(visibility_filter), iteration)
                os.makedirs(os.path.join(dataset.model_path, "data/vis_mask"), exist_ok=True)
                torch.save(visibility_filter.cpu(), os.path.join(dataset.model_path, "data/vis_mask", f"{'%05d' % iteration}.pt"))
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            if pmask is None:
                loss_mask = None
            else:
                loss_mask = pmask.reshape(image.shape[1:]).to(torch.float)

            E_u, Ll1 = l1_loss(image, gt_image, mask=loss_mask)
            lambda_dssim = opt.lambda_dssim
            loss = (1.0 - lambda_dssim) * Ll1
            if lambda_dssim > 0:
                ssim_map = ssim(image, gt_image, mask=loss_mask)
                ssim_loss = 1 - ssim_map
                loss += lambda_dssim * ssim_loss.mean()

            if aux_densify:
                R_u = render_pkg["aux_render"]
                loss += (E_u.detach() * R_u).sum()
                if lambda_dssim > 0:
                    loss += (ssim_loss.detach() * R_u).sum()

            loss = loss / len(cams)
            loss.backward()

        if aux_densify:
            new_aux_densify_grad = gaussians._aux_scalar.grad.detach()[..., 0]
            new_aux_densify_grad[new_aux_densify_grad < 0] = 0 
            aux_densify_grad = new_aux_densify_grad if aux_densify_grad is None else torch.max(aux_densify_grad, new_aux_densify_grad)
            gaussians._aux_scalar.grad = None

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
            
            training_report(opt, tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(opt, cap_max=args.cap_max, aux_grad=aux_densify_grad, add_ratio=opt.add_ratio, iteration=iteration)
                aux_densify_grad = None

            if iteration % args.vis_iteration_interval == 0:
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
                    save_image(torch.cat(figs, dim=1), os.path.join(dataset.model_path, "vis/iter_{}.png".format(iteration)))

            # Optimizer step
            if iteration < opt.iterations:
                torch.cuda.synchronize()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                
                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
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
            for idx, viewpoint in enumerate(cameras):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 5):
                    tb_writer.add_images('test' + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images('test' + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                l1_test += l1_loss(image, gt_image)[1].double()
                psnr_test += psnr(image, gt_image).mean().double()
            psnr_test /= len(cameras)
            l1_test /= len(cameras)
            if not opt.turn_off_print:
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, 'test', l1_test, psnr_test))
            if tb_writer:
                tb_writer.add_scalar('test' + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar('test' + '/loss_viewpoint - psnr', psnr_test, iteration)
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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    
    # print("\nTraining complete.")
    render_sets(lp.extract(args), op.iterations, pp.extract(args), True, False)
    evaluate([args.model_path])