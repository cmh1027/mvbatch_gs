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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "auxiliary.h"

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, int B, int S, int BR,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dpixel_depths,
		const float* dL_dpixel_trans,
		float4* dL_dmean2D,
		float* dL_dmean2D_sq,
		float* dL_dmean2D_N,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_ddepths,
		const int* mask,
		const int* batch_map,
		const int* point_index,
		const int* point_batch_index,
		const bool return_2d_grad
	);

	void preprocess(
		int P, int D, int M, int BR,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float6* cov3Ds,
		const float* view,
		const float* proj,
		const float* focal_x, 
		const float* focal_y,
		const float* tan_fovx, 
		const float* tan_fovy,
		const glm::vec3* campos,
		const float4* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_ddepth,
		float6* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		const int* mask,
		const int* point_index,
		const int* point_batch_index,
		const float low_pass
	);
}

#endif
