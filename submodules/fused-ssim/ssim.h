#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor, torch::Tensor>
fusedssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &mask
);

torch::Tensor
fusedssim_backward(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &mask,
    torch::Tensor &denom_buffer,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12,
    bool normalize_backward
);

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