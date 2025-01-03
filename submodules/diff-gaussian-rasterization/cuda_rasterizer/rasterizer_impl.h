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

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>
#include "auxiliary.h"
namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}


	struct CacheState
	{
		int* batch_num_rendered;
		int* batch_num_rendered_sums;
		bool* batch_rendered_check;
		size_t scan_size;
		char* scanning_space;
		bool* is_in_frustum;
		float* depths;
		float6* cov3D;
		float3* cov2D;
		
		static CacheState fromChunk(char*& chunk, size_t P, size_t B);
	};

	struct GeometryState
	{
		size_t scan_size;
		char* scanning_space;
		bool* clamped;
		float2* means2D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;
		int* point_index;
		int* point_batch_index;

		static GeometryState fromChunk(char*& chunk, size_t BR);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	template<typename T> 
	size_t required(size_t P1, size_t P2)
	{
		char* size = nullptr;
		T::fromChunk(size, P1, P2);
		return ((size_t)size) + 128;
	}

};