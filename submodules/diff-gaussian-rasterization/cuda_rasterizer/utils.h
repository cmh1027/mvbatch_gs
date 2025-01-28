#ifndef CUDA_RASTERIZER_UTILS_H_INCLUDED
#define CUDA_RASTERIZER_UTILS_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace UTILS
{
    void ComputeRelocation(
        int P,
        float* opacity_old,
        float* scale_old,
        int* N,
        float* binoms,
        int n_max,
        float* opacity_new,
        float* scale_new
    );

    void MakeCategoryMask(
        const int* mask,
        int H, int W, int B,
        int* output_mask
    );

    void ExtractVisiblePoints(
        int P, int B, float boundary,
        const float* orig_points,
        const float* viewmatrix,
        const float* projmatrix,
        int* visibility
    );
}

#endif