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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
#endif

#define PRINT_CUDA_ARRAY(N, type, src) \
	int temp[N]; \
	cudaMemcpy(temp, src, sizeof(type) * N, cudaMemcpyDeviceToHost); \
	for(int i=0; i<N; ++i){ \
		printf("%d ", temp[i]); \
	} \
    printf("\n"); \
	exit(0);

// #define DEBUG
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
