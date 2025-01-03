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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, p_view);
}

__global__ void savePointIndexCUDA(
	int P, int B,
    const int* batch_num_rendered_sums,
	const bool* batch_rendered_check,
	int* point_index,
	int* point_batch_index)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	int num_rendered_offset = (idx == 0) ? 0 : batch_num_rendered_sums[idx-1];
	int j=0;
	for(int i=0; i<B; ++i){
		if(batch_rendered_check[B * idx + i]){
			point_index[num_rendered_offset+j] = idx;
			point_batch_index[num_rendered_offset+j] = i;
			j += 1;
		}
	}
}

void savePointIndex(
	int P, int B,
    const int* batch_num_rendered_sums,
	const bool* batch_rendered_check,
	int* point_index,
	int* point_batch_index)
{
	savePointIndexCUDA << <(P + 255) / 256, 256 >> > (P, B, batch_num_rendered_sums, batch_rendered_check, point_index, point_batch_index);
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeysCUDA(
	int BR, int P, int B, int W,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	dim3 grid,
	const int* point_index,
	const int* point_batch_index)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= BR)
		return;
	auto point_idx = point_index[idx];
	auto batch_idx = (uint64_t)point_batch_index[idx];
	int abs_idx = point_idx * B + batch_idx;

	radii += P * batch_idx;
	// Generate no key/value pair for invisible Gaussians
	if (radii[point_idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		getRect(points_xy[idx], radii[point_idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID(2)  |  batch ID(2)   |  depth(4)  |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 48;
				key |= (batch_idx << 32) & BATCH_MASK;
				key |= *((uint32_t*)&depths[abs_idx]) & DEPTH_MASK;

				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

void duplicateWithKeys(
	int BR, int P, int B, int W,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	dim3 grid,
	const int* point_index,
	const int* point_batch_index)
{
	duplicateWithKeysCUDA << <(BR + 255) / 256, 256 >> > (
		BR, P, B, W,
		points_xy,
		depths,
		offsets,
		gaussian_keys_unsorted,
		gaussian_values_unsorted,
		radii,
		grid,
		point_index,
		point_batch_index
	);
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
/*
e.g batch=4
(A, B) => range info for tile A batch B
ranges = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2) ...]

*/
__global__ void identifyTileRangesCUDA(int L, int B, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = (key & TILE_MASK) >> 48;
	uint32_t currbatch = (key & BATCH_MASK) >> 32;
	uint32_t curridx = currtile * B + currbatch;
	if (idx == 0)
		ranges[curridx].x = 0;
	else
	{
		uint32_t prevtile = (point_list_keys[idx - 1] & TILE_MASK) >> 48;
		uint32_t prevbatch = (point_list_keys[idx - 1] & BATCH_MASK) >> 32;
		uint32_t previdx = prevtile * B + prevbatch;
		if (curridx != previdx)
		{
			ranges[previdx].y = idx;
			ranges[curridx].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[curridx].y = L;
}

void identifyTileRanges(int L, int B, uint64_t* point_list_keys, uint2* ranges)
{
	identifyTileRangesCUDA << <(L + 255) / 256, 256 >> > (
		L, B,
		point_list_keys,
		ranges
	);
	
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::CacheState CudaRasterizer::CacheState::fromChunk(char*& chunk, size_t P, size_t B)
{
	CacheState cache;
	obtain(chunk, cache.batch_num_rendered, P, 128);
	obtain(chunk, cache.batch_num_rendered_sums, P, 128);
	obtain(chunk, cache.batch_rendered_check, P*B, 128);
    cub::DeviceScan::InclusiveSum(nullptr, cache.scan_size, cache.batch_num_rendered, cache.batch_num_rendered_sums, P);
    obtain(chunk, cache.scanning_space, cache.scan_size, 128);
	obtain(chunk, cache.is_in_frustum, P*B, 128);
	obtain(chunk, cache.depths, P*B, 128);
	obtain(chunk, cache.cov3D, P, 128);
	obtain(chunk, cache.cov2D, P*B, 128);
	return cache;
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t BR)
{
	GeometryState geom;
	obtain(chunk, geom.clamped, BR * 3, 128);
	obtain(chunk, geom.means2D, BR, 128);
	obtain(chunk, geom.conic_opacity, BR, 128);
	obtain(chunk, geom.rgb, BR * 3, 128);
	obtain(chunk, geom.point_index, BR, 128);
	obtain(chunk, geom.point_batch_index, BR, 128);
	obtain(chunk, geom.tiles_touched, BR, 128);
	obtain(chunk, geom.point_offsets, BR, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, BR);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);


	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
std::tuple<int, int, float, float, float> CudaRasterizer::Rasterizer::forward(
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
	bool debug)
{
	clock_t start;
	double measureTime, preprocessTime, renderTime;

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const dim3 render_tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, S);
	const dim3 render_block(BLOCK_X * BLOCK_Y / S, 1, 1);
	assert(BLOCK_X*BLOCK_Y % S == 0);

	size_t cache_chunk_size = required<CacheState>(P, B);
	char* cache_chunkptr = cacheBuffer(cache_chunk_size);
	CacheState cacheState = CacheState::fromChunk(cache_chunkptr, P, B);
	int BR;

	TIME_CHECK(FORWARD::measureBufferSize(
		P, D, M, B,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, 
		focal_y,
		tan_fovx, 
		tan_fovy,
		tile_grid,
		mask,
		cacheState.batch_num_rendered,
		cacheState.batch_rendered_check,
		cacheState.is_in_frustum,
		cacheState.depths,
		cacheState.cov3D,
		cacheState.cov2D,
		low_pass
	), time_check, start, measureTime)


    cub::DeviceScan::InclusiveSum(cacheState.scanning_space, cacheState.scan_size, cacheState.batch_num_rendered, cacheState.batch_num_rendered_sums, P);
	cudaMemcpy(&BR, cacheState.batch_num_rendered_sums + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

	size_t chunk_size = required<GeometryState>(BR);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, BR);
	
	savePointIndex(P, B,
		cacheState.batch_num_rendered_sums, 
		cacheState.batch_rendered_check, 
		geomState.point_index, 
		geomState.point_batch_index
	);

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	TIME_CHECK(FORWARD::preprocess(
		BR, P, B, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, 
		focal_y,
		tan_fovx, 
		tan_fovy,
		cacheState.is_in_frustum,
		radii,
		geomState.means2D,
		cacheState.cov3D,
		cacheState.cov2D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		mask,
		geomState.point_index,
		geomState.point_batch_index,
		low_pass
	), time_check, start, preprocessTime)


	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, BR);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + BR - 1, sizeof(int), cudaMemcpyDeviceToHost);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys(
		BR, P, B, width,
		geomState.means2D,
		cacheState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		geomState.point_index,
		geomState.point_batch_index
	);
	
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 48 + bit
	);

	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0){
		identifyTileRanges(
			num_rendered, B,
			binningState.point_list_keys,
			imgState.ranges
		);
	}

	ERROR_CHECK
	// Let each tile blend its range of Gaussians independently in parallel


	TIME_CHECK(FORWARD::render(
		render_tile_grid, render_block,
		imgState.ranges,
		binningState.point_list,
		width, height, B, S,
		geomState.means2D,
		geomState.rgb,
		cacheState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		out_trans,
		mask,
		batch_map,
		geomState.point_batch_index
	), time_check, start, renderTime)


	return std::make_tuple(num_rendered, BR, measureTime, preprocessTime, renderTime);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
std::tuple<float, float> CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,
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
	bool debug)
{

	clock_t start;
	double preprocessTime, renderTime;

	CacheState cacheState = CacheState::fromChunk(cache_buffer, P, B);
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, BR);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
	
	assert(sizeof(int) == sizeof(uint32_t));
	cudaMemcpy(point_idx, geomState.point_index, sizeof(int)*BR, cudaMemcpyDeviceToDevice);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const dim3 render_tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, S);
	const dim3 render_block(BLOCK_X * BLOCK_Y / S, 1, 1);
	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.

	TIME_CHECK(BACKWARD::render(
		render_tile_grid, render_block,
		imgState.ranges,
		binningState.point_list,
		width, height, B, S, BR,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.rgb,
		cacheState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_depth,
		dL_dpix_trans,
		(float4*)dL_dmean2D,
		dL_dmean2D_sq,
		dL_dmean2D_N,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_ddepth,
		mask,
		batch_map,
		geomState.point_index,
		geomState.point_batch_index,
		return_2d_grad
	), time_check, start, renderTime)
	ERROR_CHECK


	TIME_CHECK(BACKWARD::preprocess(P, D, M, BR,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cacheState.cov3D,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx, 
		tan_fovy,
		(glm::vec3*)campos,
		(float4*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		(float6*)dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		mask,
		geomState.point_index,
		geomState.point_batch_index,
		low_pass
	), time_check, start, preprocessTime)
	ERROR_CHECK

	return std::make_tuple(preprocessTime, renderTime);
}
