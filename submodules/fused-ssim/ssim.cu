#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

#define G_00 0.001028380123898387f
#define G_01 0.0075987582094967365f
#define G_02 0.036000773310661316f
#define G_03 0.10936068743467331f
#define G_04 0.21300552785396576f
#define G_05 0.26601171493530273f
#define G_06 0.21300552785396576f
#define G_07 0.10936068743467331f
#define G_08 0.036000773310661316f
#define G_09 0.0075987582094967365f
#define G_10 0.001028380123898387f

// block size
#define BX 32
#define BY 32

// shared memory size
#define SX (BX + 10)
#define SSX (BX + 10)
#define SY (BY + 10)

// convolution scratchpad size
#define CX (BX)
#define CCX (BX + 0)
#define CY (BY + 10)

#define C1 0.0001
#define C2 0.0009

#ifdef DEBUG
    #define TIMEPRINT(fmt, ...) printf(fmt, ##__VA_ARGS__);
    #define PRINTLINE printf("End %s %d\n", __FILE__, __LINE__);
    #define ERROR_CHECK \
    { \
	cudaError_t err = cudaGetLastError(); \ 
	if (err != cudaSuccess) { \
		printf("CUDA Error: %s in %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(0); \
	} \
	cudaDeviceSynchronize();  \ 
	err = cudaGetLastError();   \
	if (err != cudaSuccess) { \
		printf("CUDA Error after sync: %s %s in line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(0); \
	} \
    }
#else
    #define TIMEPRINT(fmt, ...) // Do nothing if DEBUG is not defined
    #define PRINTLINE ;
    #define ERROR_CHECK ;
#endif

__device__ float get_pix_value(const float* img, const int c, const int y, const int x, const int H, const int W) {
  if (x >= W || y >= H || x < 0 || y < 0) {
    return 0.0f;
  } else {
    return img[c * H * W + y * W + x];
  }
}

__device__ float set_pix_value(float* img, float value, const int c, const int y, const int x, const int H, const int W) {
  if (x >= W || y >= H || x < 0 || y < 0) {
    return;
  } else {
    return img[c * H * W + y * W + x] = value;
  }
}

__device__ void load_into_shared(float pixels[SY][SSX], const float *inp, const int H, const int W, const int i) {
  auto block = cg::this_thread_block();
  const int start_y = block.group_index().y * BY;
  const int start_x = block.group_index().x * BX;

  const int cnt = SY * SX;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX;
      int local_x = tid % SX;
      int y = start_y + local_y;
      int x = start_x + local_x;
      float one = get_pix_value(inp, i, y - 5, x - 5, H, W);
      pixels[local_y][local_x] = one;
    }
  }
}

__device__ void multiply_shared_mem(float pix1[SY][SSX], float pix2[SY][SSX]) {
  auto block = cg::this_thread_block();
  const int cnt = SY * SX;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX;
      int local_x = tid % SX;
      float one = pix1[local_y][local_x];
      float two = pix2[local_y][local_x];
      pix1[local_y][local_x] = one * two;
    }
  }
}

__device__ inline float do_sq(float val) {
  return val * val;
}

__device__ void
flush_conv_scratch(float buf[CY][CCX]) {
  auto block = cg::this_thread_block();
  const int cnt = CY * CX;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    const int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      const int local_y = tid / CX;
      const int local_x = tid % CX;
      buf[local_y][local_x] = 0.0f;
    }
  }
}

__device__ void do_separable_conv_x(float pixels[SY][SSX], float opt[CY][CCX], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();

  int local_y = block.thread_index().y;
  int local_x = block.thread_index().x + 5;
  float val = 0.0f;

  if (sq) {
    val += G_00 * do_sq(pixels[local_y][local_x - 5]);
    val += G_01 * do_sq(pixels[local_y][local_x - 4]);
    val += G_02 * do_sq(pixels[local_y][local_x - 3]);
    val += G_03 * do_sq(pixels[local_y][local_x - 2]);
    val += G_04 * do_sq(pixels[local_y][local_x - 1]);
    val += G_05 * do_sq(pixels[local_y][local_x    ]);
    val += G_06 * do_sq(pixels[local_y][local_x + 1]);
    val += G_07 * do_sq(pixels[local_y][local_x + 2]);
    val += G_08 * do_sq(pixels[local_y][local_x + 3]);
    val += G_09 * do_sq(pixels[local_y][local_x + 4]);
    val += G_10 * do_sq(pixels[local_y][local_x + 5]);
  } else {
    val += G_00 * pixels[local_y][local_x - 5];
    val += G_01 * pixels[local_y][local_x - 4];
    val += G_02 * pixels[local_y][local_x - 3];
    val += G_03 * pixels[local_y][local_x - 2];
    val += G_04 * pixels[local_y][local_x - 1];
    val += G_05 * pixels[local_y][local_x    ];
    val += G_06 * pixels[local_y][local_x + 1];
    val += G_07 * pixels[local_y][local_x + 2];
    val += G_08 * pixels[local_y][local_x + 3];
    val += G_09 * pixels[local_y][local_x + 4];
    val += G_10 * pixels[local_y][local_x + 5];
  }
  opt[local_y][local_x] = val;

  val = 0.0f;
  local_y = block.thread_index().y + BY;
  if (local_y < SY) {
    if (sq) {
      val += G_00 * do_sq(pixels[local_y][local_x - 5]);
      val += G_01 * do_sq(pixels[local_y][local_x - 4]);
      val += G_02 * do_sq(pixels[local_y][local_x - 3]);
      val += G_03 * do_sq(pixels[local_y][local_x - 2]);
      val += G_04 * do_sq(pixels[local_y][local_x - 1]);
      val += G_05 * do_sq(pixels[local_y][local_x    ]);
      val += G_06 * do_sq(pixels[local_y][local_x + 1]);
      val += G_07 * do_sq(pixels[local_y][local_x + 2]);
      val += G_08 * do_sq(pixels[local_y][local_x + 3]);
      val += G_09 * do_sq(pixels[local_y][local_x + 4]);
      val += G_10 * do_sq(pixels[local_y][local_x + 5]);
    } else {
      val += G_00 * pixels[local_y][local_x - 5];
      val += G_01 * pixels[local_y][local_x - 4];
      val += G_02 * pixels[local_y][local_x - 3];
      val += G_03 * pixels[local_y][local_x - 2];
      val += G_04 * pixels[local_y][local_x - 1];
      val += G_05 * pixels[local_y][local_x    ];
      val += G_06 * pixels[local_y][local_x + 1];
      val += G_07 * pixels[local_y][local_x + 2];
      val += G_08 * pixels[local_y][local_x + 3];
      val += G_09 * pixels[local_y][local_x + 4];
      val += G_10 * pixels[local_y][local_x + 5];
    }
    opt[local_y][local_x] = val;
  }
}

__device__ float do_separable_conv_y(float pixels[CY][CCX], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y + 5;
  int local_x = block.thread_index().x + 5;
  float val = 0.0f;

  val += G_00 * pixels[local_y - 5][local_x];
  val += G_01 * pixels[local_y - 4][local_x];
  val += G_02 * pixels[local_y - 3][local_x];
  val += G_03 * pixels[local_y - 2][local_x];
  val += G_04 * pixels[local_y - 1][local_x];
  val += G_05 * pixels[local_y    ][local_x];
  val += G_06 * pixels[local_y + 1][local_x];
  val += G_07 * pixels[local_y + 2][local_x];
  val += G_08 * pixels[local_y + 3][local_x];
  val += G_09 * pixels[local_y + 4][local_x];
  val += G_10 * pixels[local_y + 5][local_x];

  return val;
}

__global__ void fusedssimCUDA(
  int H,
  int W,
  int CH,
  const float* img1,
  const float* img2,
  const float* mask,
  float* denom_buffer,
  float* ssim_map,
  float* dm_dmu1,
  float* dm_dsigma1_sq,
  float* dm_dsigma12
)
{
  auto block = cg::this_thread_block();
  const int batch_idx = block.group_index().z;
  img1 += batch_idx * CH * H * W;
  img2 += batch_idx * CH * H * W;
  mask += batch_idx * H * W;
  denom_buffer += batch_idx * H * W;
  ssim_map += batch_idx * CH * H * W;
  dm_dmu1 += batch_idx * CH * H * W;
  dm_dsigma1_sq += batch_idx * CH * H * W;
  dm_dsigma12 += batch_idx * CH * H * W;


  
  const int pix_y = block.group_index().y * BY + block.thread_index().y;
  const int pix_x = block.group_index().x * BX + block.thread_index().x;
  const int pix_id = pix_y * W + pix_x;
  const int num_pix = H * W;

  // shared memory that will be used to load pixels temporarily
  __shared__ float buf1[SY][SSX];
  __shared__ float buf2[SY][SSX];
  __shared__ float buf3[CY][CCX];

  load_into_shared(buf1, mask, H, W, 0);
  block.sync();
  do_separable_conv_x(buf1, buf3, H, W);
  block.sync();
  float denom = do_separable_conv_y(buf3, H, W);
  float denom_inv = (denom > 0) ? (1/denom) : 1;
  set_pix_value(denom_buffer, denom_inv, 0, pix_y, pix_x, H, W);
  block.sync();

  for (int i = 0; i < CH; ++i) {
    // load into shared
    load_into_shared(buf1, img1, H, W, i);
    block.sync();

    // calculate mu1
    do_separable_conv_x(buf1, buf3, H, W);
    block.sync();
    float mu1 = do_separable_conv_y(buf3, H, W) * denom_inv;
    block.sync();

    // calculate sigma1_sq
    do_separable_conv_x(buf1, buf3, H, W, true);
    block.sync();
    float sigma1_sq = do_separable_conv_y(buf3, H, W) * denom_inv - mu1 * mu1;
    block.sync();

    // calculate mu2
    load_into_shared(buf2, img2, H, W, i);
    block.sync();
    do_separable_conv_x(buf2, buf3, H, W);
    block.sync();
    float mu2 = do_separable_conv_y(buf3, H, W) * denom_inv;
    block.sync();

    // calculate sigma2_sq
    do_separable_conv_x(buf2, buf3, H, W, true);
    block.sync();
    float sigma2_sq = do_separable_conv_y(buf3, H, W) * denom_inv - mu2 * mu2;
    block.sync();

    // calculate sigma12
    multiply_shared_mem(buf1, buf2);
    block.sync();
    do_separable_conv_x(buf1, buf3, H, W);
    block.sync();
    float sigma12 = do_separable_conv_y(buf3, H, W) * denom_inv - mu1 * mu2;
    block.sync();

    float mu1_sq = mu1 * mu1;
    float mu2_sq = mu2 * mu2;
    float mu1_mu2 = mu1 * mu2;
    float C = (2.0f * mu1_mu2 + C1);
    float D = (2.0f * sigma12 + C2);
    float A = (mu1_sq + mu2_sq + C1);
    float B = (sigma1_sq + sigma2_sq + C2);
    float m = (C * D) / (A * B);
    if (pix_x < W && pix_y < H) {
      const int global_idx = i * num_pix + pix_id;
      ssim_map[global_idx] = m;
      if (dm_dmu1) {
        dm_dmu1[global_idx] = (
          (mu2 * 2.0f * D) / (A * B)
          -(mu2 * 2.0f * C) / (A * B)
          -(mu1 * 2.0f * C * D) / ( A * A * B)
          +(mu1 * 2.0f * C * D) / (A * B * B)
        );
        dm_dsigma1_sq[global_idx] = ((-C * D) / (A * B * B));
        dm_dsigma12[global_idx] = ((2 * C) / (A * B));
      }
    }
  }
}

__global__ void fusedssim_backwardCUDA(
  int H,
  int W,
  int CH,
  const float* img1,
  const float* img2,
  const float* mask,
  const float* denom_buffer,
  const float* dL_dmap,
  float *dL_dimg1,
  float* dm_dmu1,
  float* dm_dsigma1_sq,
  float* dm_dsigma12,
  bool normalize_backward
)
{
  auto block = cg::this_thread_block();
  const int batch_idx = block.group_index().z;
  img1 += batch_idx * CH * H * W;
  img2 += batch_idx * CH * H * W;
  mask += batch_idx * H * W;
  denom_buffer += batch_idx * H * W;
  dm_dmu1 += batch_idx * CH * H * W;
  dm_dsigma1_sq += batch_idx * CH * H * W;
  dm_dsigma12 += batch_idx * CH * H * W;
  dL_dimg1 += batch_idx * CH * H * W;
  
  const int pix_y = block.group_index().y * BY + block.thread_index().y;
  const int pix_x = block.group_index().x * BX + block.thread_index().x;
  const int pix_id = pix_y * W + pix_x;
  const int num_pix = H * W;
  bool masked = (get_pix_value(mask, 0, pix_y, pix_x, H, W) == 0);

  // shared memory that will be used to load pixels temporarily
  __shared__ float buf1[SY][SSX];
  __shared__ float buf2[SY][SSX];
  __shared__ float buf3[CY][CCX];

  for (int i = 0; i < CH; ++i) {
    float dL_dpix = 0.0f;
    float pix1 = get_pix_value(img1, i, pix_y, pix_x, H, W);
    float pix2 = get_pix_value(img2, i, pix_y, pix_x, H, W);
    load_into_shared(buf1, dL_dmap, H, W, i);
    if(normalize_backward){
      load_into_shared(buf2, denom_buffer, H, W, 0);
      block.sync();
      multiply_shared_mem(buf1, buf2);
    }
    block.sync();

    // gradient from mu1
    load_into_shared(buf2, dm_dmu1, H, W, i);
    block.sync();
    multiply_shared_mem(buf2, buf1);
    block.sync();
    do_separable_conv_x(buf2, buf3, H, W);
    block.sync();
    if(!masked)
      dL_dpix += do_separable_conv_y(buf3, H, W);

    // gradient from sigma1_sq
    load_into_shared(buf2, dm_dsigma1_sq, H, W, i);
    block.sync();
    multiply_shared_mem(buf2, buf1);
    block.sync();
    do_separable_conv_x(buf2, buf3, H, W);
    block.sync();
    if(!masked)
      dL_dpix += pix1 * 2.0f * do_separable_conv_y(buf3, H, W);

    // gradient from sigma12
    load_into_shared(buf2, dm_dsigma12, H, W, i);
    block.sync();
    multiply_shared_mem(buf2, buf1);
    block.sync();
    do_separable_conv_x(buf2, buf3, H, W);
    block.sync();
    if(!masked)
      dL_dpix += pix2 * do_separable_conv_y(buf3, H, W);

    if (pix_x < W && pix_y < H && !masked) {
      const int global_idx = i * num_pix + pix_id;
      dL_dimg1[global_idx] = dL_dpix;
    }
  }
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor, torch::Tensor>
fusedssim(
  torch::Tensor &img1, // (B, 3, H, W)
  torch::Tensor &img2,
  torch::Tensor &mask
)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
  int B = img1.size(0);
  int CH = img1.size(1);
  int H = img1.size(2);
  int W = img1.size(3);

  dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
  dim3 block(BX, BY, 1);
  
  auto float_opts = img1.options().dtype(torch::kFloat32);

  torch::Tensor target = torch::zeros_like(img1).contiguous(); // (B, 3, H, W)
  torch::Tensor denom_buffer = torch::zeros_like(mask).contiguous();
  torch::Tensor dm_dmu1 = torch::zeros_like(img1).contiguous();
  torch::Tensor dm_dsigma1_sq = torch::zeros_like(img1).contiguous();
  torch::Tensor dm_dsigma12 = torch::zeros_like(img1).contiguous();
  fusedssimCUDA<<<grid,block>>>(
    H,
    W,
    CH,
    img1.contiguous().data<float>(),
    img2.contiguous().data<float>(),
    mask.contiguous().data<float>(),
    denom_buffer.contiguous().data<float>(),
    target.contiguous().data<float>(),
    dm_dmu1.contiguous().data<float>(),
    dm_dsigma1_sq.contiguous().data<float>(),
    dm_dsigma12.contiguous().data<float>()
  );
  ERROR_CHECK
  return std::make_tuple(target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, denom_buffer);
}

torch::Tensor fusedssim_backward(
  torch::Tensor &img1,
  torch::Tensor &img2,
  torch::Tensor &mask,
  torch::Tensor &dL_dmap,
  torch::Tensor &dm_dmu1,
  torch::Tensor &dm_dsigma1_sq,
  torch::Tensor &dm_dsigma12,
  torch::Tensor &denom_buffer,
  bool normalize_backward
)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
  int B = img1.size(0);
  int CH = img1.size(1);
  int H = img1.size(2);
  int W = img1.size(3);

  torch::Tensor dL_dimg1 = torch::zeros_like(img1).contiguous();

  dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
  dim3 block(BX, BY, 1);
  fusedssim_backwardCUDA<<<grid,block>>>(
    H,
    W,
    CH,
    img1.contiguous().data<float>(),
    img2.contiguous().data<float>(),
    mask.contiguous().data<float>(),
    denom_buffer.contiguous().data<float>(),
    dL_dmap.contiguous().data<float>(),
    dL_dimg1.contiguous().data<float>(),
    dm_dmu1.contiguous().data<float>(),
    dm_dsigma1_sq.contiguous().data<float>(),
    dm_dsigma12.contiguous().data<float>(),
    normalize_backward
  );
  ERROR_CHECK
  return dL_dimg1;
}
