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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int point_idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[point_idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + point_idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float6& cov3D, const float* viewmatrix, const float low_pass)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D.x, cov3D.y, cov3D.z,
		cov3D.y, cov3D.w, cov3D.a,
		cov3D.z, cov3D.a, cov3D.b);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += low_pass;
	cov[1][1] += low_pass;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float6& cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D.x = Sigma[0][0];
	cov3D.y = Sigma[0][1];
	cov3D.z = Sigma[0][2];
	cov3D.w = Sigma[1][1];
	cov3D.a = Sigma[1][2];
	cov3D.b = Sigma[2][2];
}

__global__ void _in_frustum(
	int P, int B,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* is_in_frustum,
	float* depths
)
{
	auto idx = cg::this_grid().thread_rank();

	int batch_idx = idx / P;
	if (batch_idx >= B || idx >= B * P) return;
	idx = idx % P;
	int abs_idx = idx * B + batch_idx;

	/* initialization */
	is_in_frustum[abs_idx] = false;

	/* batch offset */
	viewmatrix += batch_idx * 16;
	projmatrix += batch_idx * 16;

	// Perform near culling, quit if outside.
	float3 p_view;
	is_in_frustum[abs_idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, p_view);
	depths[abs_idx] = p_view.z;
}

__global__ void precomputeCov3D(
	int P, 
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	float6* cov3Ds
)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) return;
	idx = idx % P;
	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds[idx]);
}

__global__ void measureBufferSizeCUDA(int P, int D, int M, int B,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float* tan_fovx, 
	const float* tan_fovy,
	const float* focal_x, 
	const float* focal_y,
	const float6* cov3Ds,
	float3* cov2Ds,
	const bool* is_in_frustum,
	const dim3 grid,
	const int* mask,
	int* batch_num_rendered,
	bool* batch_rendered_check,
	const float low_pass
)
{
	auto idx = cg::this_grid().thread_rank();

	int batch_idx = idx / P;
	if (batch_idx >= B || idx >= B * P) return;
	idx = idx % P;
	int abs_idx = idx * B + batch_idx;

	/* batch offset */
	viewmatrix += batch_idx * 16;
	projmatrix += batch_idx * 16;
	cam_pos = (glm::vec3*)((float3*)cam_pos + batch_idx);

	// Perform near culling, quit if outside.
	if (!is_in_frustum[abs_idx])
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint32_t vertial_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;

	float3 cov = computeCov2D(p_orig, focal_x[batch_idx], focal_y[batch_idx], tan_fovx[batch_idx], tan_fovy[batch_idx], cov3Ds[idx], viewmatrix, low_pass);
	cov2Ds[abs_idx] = cov;

	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	auto tiles = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
	if (tiles == 0)
		return;

	atomicAdd(batch_num_rendered + idx, 1);
	batch_rendered_check[abs_idx] = true;
}

template<int C>
__global__ void preprocessCUDA(int BR, int P, int B, int D, int M,
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
	const float* tan_fovx, 
	const float* tan_fovy,
	const float* focal_x, 
	const float* focal_y,
	const bool* is_in_frustum,
	int* radii,
	float2* points_xy_image,
	const float6* cov3Ds,
	const float3* cov2Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	const int* mask,
	const int* point_index,
	const int* point_batch_index,
	const float low_pass
)
{
	auto idx = cg::this_grid().thread_rank();

	if (idx >= BR)
		return;
	int point_idx = point_index[idx];
	int batch_idx = point_batch_index[idx];
	int abs_idx = point_idx * B + batch_idx;

	/* batch offset */
	viewmatrix += batch_idx * 16;
	projmatrix += batch_idx * 16;
	cam_pos = (glm::vec3*)((float3*)cam_pos + batch_idx);
	radii += P * batch_idx;

	// Perform near culling, quit if outside.
	if (!is_in_frustum[abs_idx])
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * point_idx], orig_points[3 * point_idx + 1], orig_points[3 * point_idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint32_t vertial_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;
	points_xy_image[idx] = point_image;

	// Compute 2D screen-space covariance matrix
	float3 cov = cov2Ds[abs_idx];

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f){
		return;
	}
		
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	auto tiles = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
	if (tiles == 0)
		return;

	glm::vec3 result = computeColorFromSH(idx, point_idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
	rgb[idx * C + 0] = result.x;
	rgb[idx * C + 1] = result.y;
	rgb[idx * C + 2] = result.z;

	// Store some useful helper data for the next steps.
	radii[point_idx] = my_radius;
	
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[point_idx] };
	tiles_touched[idx] = tiles;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t C, int BATCH>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int B,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	float* __restrict__ out_trans,
	const int* __restrict__ mask, // (PW*PH, BLOCK_X*BLOCK_Y)
	const int* __restrict__ batch_map, 
	const int* __restrict__ point_batch_index)
{
	// Identify current tile and associated min/max pixel range.
	static constexpr int THREAD_SIZE = BLOCK_SIZE / BATCH;
	static constexpr int THREAD_SIZE_COLOR = BLOCK_SIZE * C / BATCH;
	static constexpr bool NO_BATCH = BATCH == 1;

	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	auto block = cg::this_thread_block();
	uint block_x = block.group_index().x;
	uint block_y = block.group_index().y;
	uint block_id = block_y * horizontal_blocks + block_x;

	auto PH = (H + BLOCK_Y - 1) / BLOCK_Y;
	auto PW = (W + BLOCK_X - 1) / BLOCK_X;

	auto part_idx = block.group_index().z;
	auto batch_idx = batch_map[part_idx];
	auto pix_in_block_id = block_id * BLOCK_SIZE + THREAD_SIZE * part_idx + block.thread_index().x;

	bool mask_inside = pix_in_block_id < PH * PW * BLOCK_SIZE;
	int pix_in_block;
	if(mask_inside){
		if(NO_BATCH){
			pix_in_block = block.thread_index().y * W + block.thread_index().x;
		}
		else{
			pix_in_block = mask[pix_in_block_id];
		}
	}
	else{
		pix_in_block = 0;
	}
	auto pix_x_in_block = pix_in_block % BLOCK_X;
	auto pix_y_in_block = pix_in_block / BLOCK_X;

	uint2 pix_min = { block_x * BLOCK_X, block_y * BLOCK_Y };
	uint2 pix = { pix_min.x + pix_x_in_block, pix_min.y + pix_y_in_block };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint range_id = block_id * B + batch_idx;
	uint2 range = ranges[range_id];
	const int rounds = ((range.y - range.x + THREAD_SIZE - 1) / THREAD_SIZE);
	int toDo = range.y - range.x;
	
	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[THREAD_SIZE];
	__shared__ float2 collected_xy[THREAD_SIZE];
	__shared__ float4 collected_conic_opacity[THREAD_SIZE];
	__shared__ float collected_colors[THREAD_SIZE_COLOR];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float feature[C] = { 0 };
	// float D = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= THREAD_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == THREAD_SIZE)
			break;
		
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * THREAD_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * THREAD_SIZE + block.thread_rank()] = features[coll_id * C + i];
		}
		block.sync();
		
		// Iterate over current batch
		for (int j = 0; !done && j < min(THREAD_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < C; ch++)
				feature[ch] += collected_colors[ch * THREAD_SIZE + j] * alpha * T;
			// for (int ch = 0; ch < C; ch++)
			// 	feature[ch] += features[collected_id[j] * C + ch] * alpha * T;
			// D += depths[collected_id[j] * B + batch_idx] * alpha * T; 

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < C; ch++)
			out_color[ch * H * W + pix_id] = feature[ch] + T * bg_color[ch];
		// out_depth[pix_id] = D;
		// out_trans[pix_id] = T;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int B, int S,
	const float2* means2D,
	const float* colors,
	const float* depths,
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
)
{
	switch(S){
		case 1:
			renderCUDA<CHANNELS, 1> << <grid, block >> > (
				ranges, point_list, W, H, B, means2D, colors, depths, conic_opacity, final_T, n_contrib, bg_color, out_color, out_depth, out_trans, mask, batch_map, point_batch_index
			);
			break;
		case 2:
			renderCUDA<CHANNELS, 2> << <grid, block >> > (
				ranges, point_list, W, H, B, means2D, colors, depths, conic_opacity, final_T, n_contrib, bg_color, out_color, out_depth, out_trans, mask, batch_map, point_batch_index
			);
			break;
		case 4:
			renderCUDA<CHANNELS, 4> << <grid, block >> > (
				ranges, point_list, W, H, B, means2D, colors, depths, conic_opacity, final_T, n_contrib, bg_color, out_color, out_depth, out_trans, mask, batch_map, point_batch_index
			);
			break;
		case 8:
			renderCUDA<CHANNELS, 8> << <grid, block >> > (
				ranges, point_list, W, H, B, means2D, colors, depths, conic_opacity, final_T, n_contrib, bg_color, out_color, out_depth, out_trans, mask, batch_map, point_batch_index
			);
			break;
		default:
			printf("Batch size %d is not supported", S);
			exit(0);
	}

	ERROR_CHECK
}

void FORWARD::preprocess(int BR, int P, int B, int D, int M,
	const float* means3D,
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
	float2* means2D,
	const float6* cov3Ds,
	const float3* cov2Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	const int* mask,
	const int* point_index,
	const int* point_batch_index,
	const float low_pass
)
{
	preprocessCUDA<CHANNELS> << <(BR + 255) / 256, 256 >> > (
		BR, P, B, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		is_in_frustum,
		radii,
		means2D,
		cov3Ds,
		cov2Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		mask,
		point_index,
		point_batch_index,
		low_pass
	);
	ERROR_CHECK
}

void FORWARD::measureBufferSize(
	int P, int D, int M, int B,
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
)
{
	_in_frustum << <(B * P + 255) / 256, 256 >> > (
		P, B,
		orig_points,
		viewmatrix,
		projmatrix,
		is_in_frustum,
		depths
	);

	precomputeCov3D << <(P + 255) / 256, 256 >> >(
		P,
		scales,
		scale_modifier,
		rotations,
		cov3Ds
	);

	measureBufferSizeCUDA << <(B * P + 255) / 256, 256 >> > (
		P, D, M, B,
		orig_points,
		scales,
		scale_modifier,
		rotations,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		cov3Ds,
		cov2Ds,
		is_in_frustum,
		grid,
		mask,
		batch_num_rendered,
		batch_rendered_check,
		low_pass
	);
	ERROR_CHECK
}
