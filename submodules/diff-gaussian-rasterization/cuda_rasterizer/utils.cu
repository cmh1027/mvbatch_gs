#include "utils.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
__global__ void compute_relocation(
    int P, 
    float* opacity_old, 
    float* scale_old, 
    int* N, 
    float* binoms, 
    int n_max, 
    float* opacity_new, 
    float* scale_new) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= P) return;
    
    int N_idx = N[idx];
    float denom_sum = 0.0f;

    // compute new opacity
    opacity_new[idx] = 1.0f - powf(1.0f - opacity_old[idx], 1.0f / N_idx);
    
    // compute new scale
    for (int i = 1; i <= N_idx; ++i) {
        for (int k = 0; k <= (i-1); ++k) {
            float bin_coeff = binoms[(i-1) * n_max + k];
            float term = (pow(-1, k) / sqrt(k + 1)) * pow(opacity_new[idx], k + 1);
            denom_sum += (bin_coeff * term);
        }
    }
    float coeff = (opacity_old[idx] / denom_sum);
    for (int i = 0; i < 3; ++i)
        scale_new[idx * 3 + i] = coeff * scale_old[idx * 3 + i];
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
make_category_mask(
    const int* mask, // (PW*PH, BLOCK_X*BLOCK_Y)
    const int* batch_map,
    int H, int W, int B,
    int* output_mask)
{
	// Identify current tile and associated min/max pixel range.
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	auto block = cg::this_thread_block();
	uint block_x = block.group_index().x;
	uint block_y = block.group_index().y;
	auto block_id = block_y * horizontal_blocks + block_x;

	auto PH = (H + BLOCK_Y - 1) / BLOCK_Y;
	auto PW = (W + BLOCK_X - 1) / BLOCK_X;

	auto pix_in_block_id = block_id * BLOCK_SIZE + block.thread_index().x;
	auto batch_idx = block.thread_index().x / (BLOCK_SIZE / B);

	bool mask_inside = pix_in_block_id < PH * PW * BLOCK_SIZE;
    if(!mask_inside) return;

	int pix_in_block = mask[pix_in_block_id];
	auto pix_x_in_block = pix_in_block % BLOCK_X;
	auto pix_y_in_block = pix_in_block / BLOCK_X;

	uint2 pix_min = { block_x * BLOCK_X, block_y * BLOCK_Y };
	uint2 pix = { pix_min.x + pix_x_in_block, pix_min.y + pix_y_in_block };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };
    bool inside = pix.x < W && pix.y < H;
    if(!inside) return;
    output_mask[pix_id] = batch_map[batch_idx];
}


void UTILS::ComputeRelocation(
    int P,
    float* opacity_old,
    float* scale_old,
    int* N,
    float* binoms,
    int n_max,
    float* opacity_new,
    float* scale_new)
{
	int num_blocks = (P + 255) / 256;
	dim3 block(256, 1, 1);
	dim3 grid(num_blocks, 1, 1);
	compute_relocation<<<grid, block>>>(P, opacity_old, scale_old, N, binoms, n_max, opacity_new, scale_new);
}

void UTILS::MakeCategoryMask(
    const int* mask,
    const int* batch_map,
    int H, int W, int B,
    int* output_mask
)
{
	dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X * BLOCK_Y, 1, 1);
	make_category_mask<<<grid, block>>>(mask, batch_map, H, W, B, output_mask);
    ERROR_CHECK
}