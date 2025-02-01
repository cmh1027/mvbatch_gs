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
from utils.loss_utils import pixel_loss, ssim
from lpipsPyTorch import lpips
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, draw_graph, draw_two_graphs, compute_pts_func
from tqdm import tqdm
from utils.image_utils import psnr, apply_depth_colormap, pcoef_freq
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
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
from diff_gaussian_rasterization import make_category_mask, extract_visible_points
from math import ceil

def training(dataset, opt, pipe, args):
	saving_iterations = args.save_iterations
	checkpoint_iterations = args.checkpoint_iterations
	checkpoint = args.start_checkpoint
	if args.test_iteration_interval is not None:
		testing_iterations = list(range(args.test_iteration_interval, opt.iterations+1, args.test_iteration_interval))
	else:
		testing_iterations = args.test_iterations

	if opt.gs_type == "original":
		dataset.init_scale = 1
		opt.densify_until_iter = 25000
		predictable_growth_degree = opt.predictable_growth_degree_3dgs
	else:
		predictable_growth_degree = opt.predictable_growth_degree_mcmc

	if opt.predictable_growth:
		print(f"Predictable growth degree : {predictable_growth_degree}")
	if opt.cap_max == -1:
		print("Please specify the maximum number of Gaussians using --cap_max.")
		exit()

	first_iter = 0
	tb_writer = prepare_output_and_logger(dataset)
	gaussians = GaussianModel(dataset.sh_degree)
	load_iter = -1 if dataset.load_iter else None
	scene = Scene(dataset, gaussians, load_iteration=load_iter)
	gaussians.training_setup(opt, num_cams=len(scene.getTrainCameras()))
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

	assert opt.mask_height == opt.mask_width
	partial_height = (H + opt.mask_height - 1) // opt.mask_height
	partial_width = (W + opt.mask_width - 1) // opt.mask_width

	print(f"Image ({H} x {W} = {H * W})")
	print(f"Tiles ({partial_height} x {partial_width})")
	print(f"tile size {opt.mask_height} x {opt.mask_width}")

	from_iter = opt.densify_from_iter
	num_pts_func = compute_pts_func(opt.cap_max, gaussians.num_pts, (opt.densify_until_iter - from_iter) // opt.densification_interval, predictable_growth_degree)

	start_time = time.time()

	for iteration in range(first_iter, opt.iterations + 1): 
		if iteration == 2000:
			breakpoint()
		gt_images = []
		xyz_lr = gaussians.update_learning_rate(iteration)

		# Every 1000 its we increase the levels of SH up to a maximum degree
		if iteration % 1000 == 0:
			gaussians.oneupSHdegree()

		# Pick a random Camera
		viewpoints = scene.getTrainCameras().copy()
		if not viewpoint_stack:
			viewpoint_stack = list(range(len(scene.getTrainCameras())))
		cam_idx = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

		if opt.batch_size == 1:
			cam_idxs = torch.tensor([cam_idx], device=torch.device('cuda'))
		else:
			cam_idxs = scene.sample_cameras(cam_idx, N=opt.batch_size)

		cam_idxs.clamp_(max=len(viewpoints)-1)
		cams = [viewpoints[idx] for idx in cam_idxs]

		bg = torch.rand((3), device="cuda") if opt.random_background else background

		pmask = torch.sort(torch.rand(partial_height * partial_width, opt.mask_height * opt.mask_width, device=torch.device('cuda'))).indices.to(torch.int32)

		kwargs = {
			"mask" : pmask,
			"grad_sep": opt.grad_sep,
			"time_check": opt.time_check
		} 
		if opt.time_check:
			torch.cuda.synchronize()
			start = time.time()
		render_pkg = render(cams, gaussians, pipe, bg, **kwargs)
		if opt.time_check:
			torch.cuda.synchronize()
			forward_time = time.time() - start

		(image, depth, residual_trans, viewspace_point_tensor, radii, gaussian_visibility, log_buffer) = (
			render_pkg["render"][:3, ...], 
			render_pkg["depth"],
			render_pkg["residual_trans"],
			render_pkg["viewspace_points"], 
			render_pkg["radii"],
			render_pkg["gaussian_visibility"],
			render_pkg["log_buffer"]
		)

		if iteration <= args.update_visibility[-1]:
			gaussians.update_gaussian_visibility(gaussian_visibility, cam_idxs) 
		if iteration in args.update_visibility:
			scene.compute_viewpoint_distance(gaussians.gaussian_visibility)
			gaussians.reset_gaussian_visibility(purge=(iteration==args.update_visibility[-1]))

		if len(cams) > 1:
			gt_images = torch.stack([cam.original_image.cuda() for cam in cams]) # (3, H, W)
			collage_mask = make_category_mask(pmask, H, W, opt.batch_size).to(torch.int64)
			collage_mask = collage_mask.unsqueeze(0).repeat(3,1,1)
			collage_gt = torch.gather(gt_images, 0, collage_mask.unsqueeze(0)).squeeze(0)
			Ll = pixel_loss(image, collage_gt, ltype=opt.loss_type)
			loss = (1.0 - opt.lambda_dssim) * Ll
			collage_mask_binary = torch.zeros_like(collage_mask[0:1]).repeat(opt.batch_size, 1, 1).float()
			collage_mask_binary.scatter_add_(0, collage_mask[0:1], torch.ones_like(collage_mask_binary)) # (B, H, W)
			if opt.lambda_dssim > 0:
				image_separated = collage_mask_binary.unsqueeze(1) * image.unsqueeze(0) # (B, C, H, W)
				gt_separated = collage_mask_binary.unsqueeze(1) * collage_gt.unsqueeze(0)  # (B, C, H, W)
				ssim_map = ssim(image_separated, gt_separated, mask=collage_mask_binary)
				loss += opt.lambda_dssim * (1 - ssim_map).sum(dim=0).mean() 
		else:
			gt_image = cams[0].original_image
			Ll = pixel_loss(image, gt_image, ltype=opt.loss_type)
			loss = (1.0 - opt.lambda_dssim) * Ll
			if opt.lambda_dssim > 0:
				loss += opt.lambda_dssim * (1 - ssim(image, gt_image)).mean()
			
		#########################
		if not opt.evaluate_time:
			if iteration % 100 == 0:
				tb_writer.add_scalar(f'train/num_points', len(gaussians._xyz), iteration)
				tb_writer.add_scalar(f'train/R', log_buffer["R"], iteration)
				tb_writer.add_scalar(f'train/BR', log_buffer["BR"], iteration)

		reg_loss = torch.tensor(0., requires_grad=True)

		if opt.gs_type == "mcmc":
			reg_loss = reg_loss + opt.opacity_reg * gaussians.get_opacity.mean() 
			reg_loss = reg_loss + opt.scale_reg * gaussians.get_scaling.mean()

		total_loss = loss + reg_loss

		if opt.time_check:
			torch.cuda.synchronize()
			start = time.time()
		total_loss.backward()
		if opt.time_check:
			torch.cuda.synchronize()
			backward_time = time.time() - start
			

		if opt.time_check and iteration % 10 == 0:
			tb_writer.add_scalar(f'time/forward/measure', log_buffer["forward_measureTime"], iteration)
			tb_writer.add_scalar(f'time/forward/preprocess', log_buffer["forward_preprocessTime"], iteration)
			tb_writer.add_scalar(f'time/forward/render', log_buffer["forward_renderTime"], iteration)
			tb_writer.add_scalar(f'time/backward/preprocess', log_buffer["backward_preprocessTime"], iteration)
			tb_writer.add_scalar(f'time/backward/render', log_buffer["backward_renderTime"], iteration)
			tb_writer.add_scalar(f'time/total/forward', forward_time, iteration)
			tb_writer.add_scalar(f'time/total/backward', backward_time, iteration)


		with torch.no_grad():
			# Progress bar
			ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
			if iteration % 10 == 0:
				postfix = {
					"Loss": f"{ema_loss_for_log:.{4}f}",
					"num_pts": len(gaussians._xyz),
					"batch": opt.batch_size
				}
				progress_bar.set_postfix(postfix)
				progress_bar.update(10)
			if iteration == opt.iterations:
				progress_bar.close()
			# Log and save
			if not opt.evaluate_time:
				training_report(opt, tb_writer, iteration, Ll, loss, testing_iterations, scene, render, (pipe, background))

			if (iteration in saving_iterations and not opt.evaluate_time):
				print("\n[ITER {}] Saving Gaussians".format(iteration))
				scene.save(iteration)

			if iteration < opt.densify_until_iter:
				if opt.gs_type == "original":
					# radii : (B, N)
					max_radii = radii.max(dim=0).values
					mask = max_radii > 0
					
					gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], max_radii[mask])
					vs_clone = viewspace_point_tensor.grad[..., 0:1]
					vs_split = viewspace_point_tensor.grad[..., 1:2]
					denom = (radii > 0).sum(dim=0)[..., None] / len(cams)
					gaussians.add_densification_stats(vs_clone, vs_split, mask, denom)

				if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
					dense_step = (iteration - from_iter) // opt.densification_interval
					next_pts_count = num_pts_func(dense_step)

					if opt.gs_type == "original":
						size_threshold = 20 if iteration > opt.opacity_reset_interval else None
						
						pts_count = min(next_pts_count - gaussians.num_pts, gaussians.num_pts)
						gaussians.densify(tb_writer, opt, scene.cameras_extent, pts_count, iteration)
						if iteration % opt.prune_interval == 0:
							gaussians.prune(tb_writer, opt, opt.prune_threshold, scene.cameras_extent, size_threshold)
							if (iteration - opt.prune_interval) % opt.opacity_reset_interval == 0:
								from_iter = iteration
								num_pts_func = compute_pts_func(opt.cap_max, gaussians.num_pts, (opt.densify_until_iter - from_iter) // opt.densification_interval, predictable_growth_degree)


					elif opt.gs_type == "mcmc":
						dead_mask = (gaussians.get_opacity <= opt.prune_threshold).squeeze(-1)
						gaussians.relocate_gs(opt, dead_mask=dead_mask)
						if not opt.evaluate_time:
							tb_writer.add_scalar(f'train/dead_gaussians', dead_mask.sum().item(), iteration)
						if opt.predictable_growth:
							add_ratio = next_pts_count / gaussians.num_pts - 1
						else:
							add_ratio = opt.add_ratio
						gaussians.add_new_gs(opt, tb_writer, iteration=iteration, cap_max=opt.cap_max, add_ratio=add_ratio)

				if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
					if opt.gs_type == "original":
						gaussians.reset_opacity(opt.opacity_reset_value)


			if iteration % args.vis_iteration_interval == 0 and not opt.evaluate_time:
				os.makedirs(os.path.join(dataset.model_path, "vis"), exist_ok=True)
				with torch.no_grad():
					render_pkg = render(viewpoints[cam_idx], gaussians, pipe, bg)
					image = render_pkg["render"][:3, ...]
					depth = render_pkg["depth"] 
					gt_image = viewpoints[cam_idx].original_image
					aux_map = depth
					beta_map = torch.zeros_like(depth)
					pred = torch.cat([image, apply_depth_colormap(aux_map.permute(1, 2, 0))], dim=-1)
					gt = torch.cat([gt_image, apply_depth_colormap(beta_map.permute(1, 2, 0))], dim=-1)
					figs = [pred, gt]
					save_image(torch.cat(figs, dim=1), os.path.join(dataset.model_path, f"vis/iter_{iteration}.png"))

			if opt.log_batch and iteration % opt.log_batch_interval == 0 and not opt.evaluate_time:
				os.makedirs(os.path.join(dataset.model_path, "batch"), exist_ok=True)
				with torch.no_grad():
					for i, image in enumerate(gt_images.unbind(dim=0)):
						os.makedirs(os.path.join(dataset.model_path, f"batch/{'%05d' % iteration}"), exist_ok=True)
						save_image(image, os.path.join(dataset.model_path, f"batch/{'%05d' % iteration}/{'%05d' % i}.png"))

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

	end_time = time.time()
	with open(os.path.join(dataset.model_path, "info.txt"), "w") as f:
		f.write(f"num_points : {gaussians.num_pts}")

	if opt.evaluate_time:
		elasped_sec = end_time - start_time
		with open(os.path.join(dataset.model_path, "elapsed.txt"), "w") as f:
			time_str = str(timedelta(seconds=elasped_sec))
			f.write(time_str)
			print(f"Elapsed time : {time_str}")
	if opt.evaluate_time:
		print("\n[ITER {}] Saving Gaussians".format(iteration))
		scene.save(iteration)
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

def training_report(opt, tb_writer, iteration, Ll, loss, testing_iterations, scene : Scene, renderFunc, renderArgs, **renderKwargs):
	if tb_writer:
		tb_writer.add_scalar(f'train_loss_patches/{opt.loss_type}_loss', Ll.item(), iteration)
		tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

	# Report test and samples of training set
	with torch.no_grad():
		if iteration in testing_iterations:
			train_cameras = scene.getTrainCameras()
			test_cameras = scene.getTestCameras()
			train_cameras = train_cameras[::len(train_cameras) // len(test_cameras)]
			for mode, cameras in [('train', train_cameras), ('test', test_cameras)]:
				l1_test = 0.0
				psnr_test = 0.0
				ssim_test = 0.0
				lpips_test = 0.0
				pcoef_freq_test = 0.0
				pcoef_low_freq_test = 0.0
				pcoef_high_freq_test = 0.0
				for idx, viewpoint in tqdm(enumerate(cameras), desc=f"Evaluating {mode}...", leave=False, total=len(cameras)):
					image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, **renderKwargs)["render"], 0.0, 1.0)
					gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
					if tb_writer and (idx < 5) and mode == 'test':
						tb_writer.add_images('test' + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
						if iteration == testing_iterations[0]:
							tb_writer.add_images('test' + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
					l1_test += pixel_loss(image, gt_image, ltype=opt.loss_type).double()


					psnr_test += psnr(image, gt_image).mean().double()
					if opt.full_evaluate:
						ssim_test += ssim(image, gt_image).mean().double()
						lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
						pcoef_freq_, pcoef_freq_low_, pcoef_freq_high_ = pcoef_freq(image, gt_image)
						pcoef_freq_test += pcoef_freq_
						pcoef_low_freq_test += pcoef_freq_low_
						pcoef_high_freq_test += pcoef_freq_high_

				psnr_test /= len(cameras)
				l1_test /= len(cameras)
				if opt.full_evaluate:
					ssim_test /= len(cameras)
					lpips_test /= len(cameras)
					pcoef_freq_test /= len(cameras)
					pcoef_low_freq_test /= len(cameras)
					pcoef_high_freq_test /= len(cameras)


				print(f"\n[ITER {iteration}] Evaluating {mode}: L1 {'%.5f' % l1_test} PSNR {'%.4f' % psnr_test} SSIM {'%.5f' % ssim_test} LPIPS {'%.5f' % lpips_test}")
				if tb_writer:
					tb_writer.add_scalar(mode + f'/loss_viewpoint - {opt.loss_type}_loss', l1_test, iteration)
					tb_writer.add_scalar(mode + '/loss_viewpoint - psnr', psnr_test, iteration)
					if opt.full_evaluate:
						tb_writer.add_scalar(mode + '/loss_viewpoint - ssim', ssim_test, iteration)
						tb_writer.add_scalar(mode + '/loss_viewpoint - lpips', lpips_test, iteration)

					if mode == 'test':
						if opt.full_evaluate:
							tb_writer.add_scalar('test' + '/loss_viewpoint - pcoef_freq', pcoef_freq_test, iteration)
							tb_writer.add_scalar('test' + '/loss_viewpoint - pcoef_freq_low', pcoef_low_freq_test, iteration)
							tb_writer.add_scalar('test' + '/loss_viewpoint - pcoef_freq_high', pcoef_high_freq_test, iteration)
						tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

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
	parser.add_argument("--render_iter", type=int)
	parser.add_argument("--override_cap_max", type=int)
	parser.add_argument("--override_degree_3dgs", type=float)
	parser.add_argument("--override_degree_mcmc", type=float)
	parser.add_argument("--update_visibility", nargs="+", type=int, default=[1500,4500,7500])


	args = parser.parse_args(sys.argv[1:])
	if args.config is not None:
		# Load the configuration file
		config = load_config(args.config)
		# Set the configuration parameters on args, if they are not already set by command line arguments
		for key, value in config.items():
			setattr(args, key, value)
	if args.override_cap_max is not None:
		args.cap_max = args.override_cap_max
	if args.override_degree_3dgs is not None:
		args.predictable_growth_degree_3dgs = args.override_degree_3dgs
	if args.override_degree_mcmc is not None:
		args.predictable_growth_degree_mcmc = args.override_degree_mcmc
	
	args.save_iterations.append(args.iterations)
	
	print("Optimizing " + args.model_path)

	# Initialize system state (RNG)
	safe_state(args.quiet)

	# Start GUI server, configure and run training
	# network_gui.init(args.ip, args.port)
	torch.autograd.set_detect_anomaly(args.detect_anomaly)
	training(lp.extract(args), op.extract(args), pp.extract(args), args)
	render_iter = op.iterations if args.render_iter is None else args.render_iter
	render_sets(lp.extract(args), render_iter, pp.extract(args), True, False)
	evaluate([args.model_path]) 


