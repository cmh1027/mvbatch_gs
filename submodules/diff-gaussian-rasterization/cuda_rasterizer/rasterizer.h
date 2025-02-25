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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static std::tuple<int, int, float, float, float> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> cacheBuffer,
			const int P, int D, int M, int B, int S,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float* focal_x, 
			const float* focal_y,
			const float* tan_fovx, 
			const float* tan_fovy,
			float* out_color,
			float* out_depth,
			float* out_trans,
			int* radii,
			const int* mask,
			const int* batch_map,
			const float low_pass,
			const bool time_check,
			bool debug = false);

		static std::tuple<float, float> backward(
			const int P, int D, int M, int B, int S, int R, int BR,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float* focal_x, 
			const float* focal_y,
			const float* tan_fovx, 
			const float* tan_fovy,
			const int* radii,
			char* cache_buffer,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dpix_depth,
			const float* dL_dpix_trans,
			float* dL_dmean2D,
			float* dL_dmean2D_sq,
			float* dL_dmean2D_N,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_ddepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			const int* mask,
			const int* batch_map,
			int* point_idx,
			const float low_pass,
			const bool return_2d_grad,
			const bool time_check,
			bool debug);
	};
};

#endif
