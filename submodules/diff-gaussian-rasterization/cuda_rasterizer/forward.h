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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "auxiliary.h"

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.

	void measureBufferSize(int P, int D, int M, int B,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float* focal_x, 
		const float* focal_y,
		const float* tan_fovx, 
		const float* tan_fovy,
		const dim3 grid,
		const int* mask,
		int* batch_num_rendered,
		bool* batch_rendered_check,
		bool* is_in_frustum,
		float* depths,
		float6* cov3Ds,
		float3* cov2Ds,
		const float low_pass
	);

	void preprocess(int BR, int P, int B, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float* focal_x, 
		const float* focal_y,
		const float* tan_fovx, 
		const float* tan_fovy,
		const bool* is_in_frustum,
		int* radii,
		float2* points_xy_image,
		const float6* cov3Ds,
		const float3* cov2Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		const int* mask,
		const int* point_index,
		const int* point_batch_index,
		const float low_pass
	);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, int B, int S,
		const float2* points_xy_image,
		const float* features,
		const float* depth,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_depth,
		float* out_trans,
		const int* mask,
		const int* batch_map,
		const int* point_batch_index
	);
}


#endif
