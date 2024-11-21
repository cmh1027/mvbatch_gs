/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float, float>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& tan_fovx, 
	const torch::Tensor& tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	torch::Tensor& mask,
	const float low_pass,
	const bool time_check,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const torch::Tensor& tan_fovx, 
	const torch::Tensor& tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& dL_dout_trans,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& cacheBuffer,
	const torch::Tensor& geomBuffer,
	const int R,
	const int BR,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	torch::Tensor& mask,
	const float low_pass,
	const bool grad_sep,
	const bool return_2d_grad,
	const bool time_check,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
		torch::Tensor& opacity_old,
		torch::Tensor& scale_old,
		torch::Tensor& N,
		torch::Tensor& binoms,
		const int n_max);

torch::Tensor MakeCategoryMaskCUDA(
	torch::Tensor& mask,
	int H, int W, int B
);