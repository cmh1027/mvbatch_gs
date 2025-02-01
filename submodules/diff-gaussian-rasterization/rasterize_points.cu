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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/utils.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		t.fill_(0);
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const int HS,
	const torch::Tensor& visibility_mapping, // (P, 1) => max values HS
	const bool write_visibility,
	const bool time_check,
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	auto PH = (image_height + BLOCK_Y - 1) / BLOCK_Y;
	auto PW = (image_width + BLOCK_X - 1) / BLOCK_X;
	if(mask.contiguous().data<int>() == nullptr){
		mask = torch::arange(BLOCK_X*BLOCK_Y, means3D.options().dtype(torch::kInt32)).unsqueeze(0).repeat({PH*PW, 1}); // (PH*PW, BLOCK_X * BLOCK_Y)
	}
	assert(mask.size(0) == PH*PW);
	assert(mask.size(1) == BLOCK_X * BLOCK_Y);
	assert(BLOCK_X == BLOCK_Y);

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;
	const int B = viewmatrix.size(0);
	assert(BLOCK_X * BLOCK_Y % B == 0);

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor out_trans = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({B, P}, 0, means3D.options().dtype(torch::kInt32));
	
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor cacheBuffer = torch::empty({0}, options.device(device));
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

	std::function<char*(size_t)> cacheFunc = resizeFunctional(cacheBuffer);
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	torch::Tensor focal_y = H / (2.0f * tan_fovy);
	torch::Tensor focal_x = W / (2.0f * tan_fovx);

	torch::Tensor gaussian_visibility = torch::full({B, HS}, 0, means3D.options().dtype(torch::kInt32));

	int rendered = 0;
	int batch_rendered = 0;
	double measureTime, preprocessTime, renderTime;
	if(P != 0)
	{
		int M = 0;
		if(sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		auto returned = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			cacheFunc,
			P, degree, M, B, HS,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			opacity.contiguous().data<float>(), 
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			focal_x.contiguous().data<float>(),
			focal_y.contiguous().data<float>(),
			tan_fovx.contiguous().data<float>(),
			tan_fovy.contiguous().data<float>(),
			out_color.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			out_trans.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			gaussian_visibility.contiguous().data<int>(),
			visibility_mapping.contiguous().data<int>(),
			write_visibility,
			mask.contiguous().data<int>(),
			time_check,
			debug
		);
		rendered = std::get<0>(returned);
		batch_rendered = std::get<1>(returned);
		measureTime = std::get<2>(returned);
		preprocessTime = std::get<3>(returned);
		renderTime = std::get<4>(returned);
	}
	return std::make_tuple(rendered, batch_rendered, out_color, out_depth, out_trans, radii, cacheBuffer, geomBuffer, binningBuffer, imgBuffer, gaussian_visibility, measureTime, preprocessTime, renderTime);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float>
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
	const torch::Tensor& mask,
	const int grad_sep,
	const bool time_check,
	const bool debug)
{
	const int B = viewmatrix.size(0);
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	torch::Tensor focal_y = H / (2.0f * tan_fovy);
	torch::Tensor focal_x = W / (2.0f * tan_fovx);

	int M = 0;
	if(sh.size(0) != 0)
	{	
		M = sh.size(1);
	}


	torch::Tensor dL_dmeans2D = torch::zeros({BR, 2}, means3D.options());
	torch::Tensor dL_dmeans2D_sq = torch::zeros({BR, 1}, means3D.options());
	
	torch::Tensor dL_dcolors = torch::zeros({BR, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_ddepths = torch::zeros({BR, 1}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({BR, 2, 2}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({BR, 6}, means3D.options());

	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor point_idx = torch::zeros({BR, 1}, means3D.options().dtype(torch::kInt32));

	torch::Tensor dL_dmeans3D = torch::zeros({BR, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({BR, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({BR, 4}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({BR, M*3}, means3D.options());

	torch::Tensor dL_dmeans2D_sum = torch::zeros({P, 2}, means3D.options());
	torch::Tensor dL_dmeans3D_sum = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dscales_sum = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations_sum = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dsh_sum = torch::zeros({P, M*3}, means3D.options());

	double preprocessTime, renderTime;
	if(BR != 0)
	{  
		auto returned = CudaRasterizer::Rasterizer::backward(P, degree, M, B, R, BR,
		background.contiguous().data<float>(),
		W, H, 
		means3D.contiguous().data<float>(),
		sh.contiguous().data<float>(),
		scales.data_ptr<float>(),
		scale_modifier,
		rotations.data_ptr<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		focal_x.contiguous().data<float>(),
		focal_y.contiguous().data<float>(),
		tan_fovx.contiguous().data<float>(),
		tan_fovy.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		reinterpret_cast<char*>(cacheBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		dL_dout_color.contiguous().data<float>(),
		dL_dout_depth.contiguous().data<float>(),
		dL_dout_trans.contiguous().data<float>(),
		dL_dmeans2D.contiguous().data<float>(),
		dL_dmeans2D_sq.contiguous().data<float>(),
		dL_dconic.contiguous().data<float>(),  
		dL_dopacity.contiguous().data<float>(),
		dL_dcolors.contiguous().data<float>(),
		dL_ddepths.contiguous().data<float>(),
		dL_dmeans3D.contiguous().data<float>(),
		dL_dcov3D.contiguous().data<float>(),
		dL_dsh.contiguous().data<float>(),
		dL_dscales.contiguous().data<float>(),
		dL_drotations.contiguous().data<float>(),
		mask.contiguous().data<int>(),
		point_idx.contiguous().data<int>(),
		time_check,
		debug);
	
		preprocessTime = std::get<0>(returned);
		renderTime = std::get<1>(returned);

		point_idx = point_idx.to(torch::kInt64);

		dL_dmeans3D_sum.scatter_add_(0, point_idx.expand({-1, 3}), dL_dmeans3D);
		dL_dscales_sum.scatter_add_(0, point_idx.expand({-1, 3}), dL_dscales);
		dL_drotations_sum.scatter_add_(0, point_idx.expand({-1, 4}), dL_drotations);
		dL_dsh_sum.scatter_add_(0, point_idx.expand({-1, M*3}), dL_dsh);
		dL_dsh_sum = dL_dsh_sum.reshape({P, M, 3});

		torch::Tensor dL_dmeans2D_clone = torch::empty({0}, means3D.options());
		torch::Tensor dL_dmeans2D_split = torch::empty({0}, means3D.options());
		switch(grad_sep){
			case 0:
				dL_dmeans2D_clone = torch::zeros({P, 2}, means3D.options());
				dL_dmeans2D_split = torch::zeros({P, 2}, means3D.options());
				dL_dmeans2D_clone.scatter_add_(0, point_idx.expand({-1, 2}), dL_dmeans2D.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}));
				dL_dmeans2D_split.scatter_add_(0, point_idx.expand({-1, 2}), dL_dmeans2D.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}));
				dL_dmeans2D_sum = torch::cat({dL_dmeans2D_clone.norm(2, -1, true), dL_dmeans2D_split.norm(2, -1, true)}, -1);
				break;
			case 1:
				dL_dmeans2D_clone = torch::zeros({P, 1}, means3D.options());
				dL_dmeans2D_split = torch::zeros({P, 1}, means3D.options());
				dL_dmeans2D_clone.scatter_add_(0, point_idx.expand({-1, 1}), dL_dmeans2D.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}).norm(2, -1, true));
				dL_dmeans2D_split.scatter_add_(0, point_idx.expand({-1, 1}), dL_dmeans2D_sq);
				dL_dmeans2D_sum = torch::cat({dL_dmeans2D_clone,dL_dmeans2D_split}, -1);
				break;
			case 2:
				dL_dmeans2D_clone = torch::zeros({P, 1}, means3D.options());
				dL_dmeans2D_split = torch::zeros({P, 1}, means3D.options());
				dL_dmeans2D_clone.scatter_add_(0, point_idx.expand({-1, 1}), dL_dmeans2D_sq);
				dL_dmeans2D_split.scatter_add_(0, point_idx.expand({-1, 1}), dL_dmeans2D.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}).norm(2, -1, true));
				dL_dmeans2D_sum = torch::cat({dL_dmeans2D_clone,dL_dmeans2D_split}, -1);
				break;
			case 3:
				dL_dmeans2D_clone = torch::zeros({P, 1}, means3D.options());
				dL_dmeans2D_split = torch::zeros({P, 1}, means3D.options());
				dL_dmeans2D_clone.scatter_add_(0, point_idx.expand({-1, 1}), dL_dmeans2D_sq);
				dL_dmeans2D_split.scatter_add_(0, point_idx.expand({-1, 1}), dL_dmeans2D_sq);
				dL_dmeans2D_sum = torch::cat({dL_dmeans2D_clone,dL_dmeans2D_split}, -1);
				break;
			case 4:
				dL_dmeans2D_clone = torch::zeros({P, 1}, means3D.options());
				dL_dmeans2D_split = torch::zeros({P, 1}, means3D.options());
				dL_dmeans2D_clone.scatter_add_(0, point_idx.expand({-1, 1}), dL_dmeans2D.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}).norm(2, -1, true));
				dL_dmeans2D_split.scatter_add_(0, point_idx.expand({-1, 1}), dL_dmeans2D.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}).norm(2, -1, true));
				dL_dmeans2D_sum = torch::cat({dL_dmeans2D_clone,dL_dmeans2D_split}, -1);
				break;
			default:
				printf("Invalid gradient separation\n");
				exit(0);
		}
	}
	ERROR_CHECK
  	return std::make_tuple(dL_dmeans2D_sum, dL_dopacity, dL_dmeans3D_sum, dL_dsh_sum, dL_dscales_sum, dL_drotations_sum, preprocessTime, renderTime);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix
)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);

}

torch::Tensor MakeCategoryMaskCUDA(
	torch::Tensor& mask,
	int H, int W, int B
)
{
	auto int_ops = mask.options().dtype(torch::kInt32);
	torch::Tensor category_mask = torch::full({H, W}, 0.0, int_ops);


	UTILS::MakeCategoryMask(
		mask.contiguous().data<int>(),
		H, W, B,
		category_mask.contiguous().data<int>()
	);


	return category_mask;
}

torch::Tensor ExtractVisiblePointsCUDA(
	const torch::Tensor& orig_points,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	float boundary
)
{
	const int P = orig_points.size(0);
	const int B = viewmatrix.size(0);

	auto int_ops = orig_points.options().dtype(torch::kInt32);
	torch::Tensor visibility = torch::full({B, P}, 0.0, int_ops);

	UTILS::ExtractVisiblePoints(
		P, B, boundary,
		orig_points.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(), 
		visibility.contiguous().data<int>()
	);

	return visibility;
}