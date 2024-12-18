#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
    torch::Tensor &pred,
    torch::Tensor &gt,
    torch::Tensor &mask,
    bool normalize
);

torch::Tensor
fusedssim_backward(
    torch::Tensor &pred,
    torch::Tensor &gt,
    torch::Tensor &mask,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
);
