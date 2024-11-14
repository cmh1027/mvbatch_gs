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
