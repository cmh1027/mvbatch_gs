#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor, torch::Tensor>
fusedssim(
    float C1,
    float C2,
    int mask_size,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &mask,
    torch::Tensor &ssim_buffer,
    bool train
);

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &mask,
    torch::Tensor &denom_buffer,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
);
