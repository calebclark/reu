/*
 ============================================================================
 Name        : GScuda.cu
 Author      : caleb
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <cub/cub.cuh>
using namespace cub;
// I ASSUME THROUGHOUT THAT sizeof(int) = 4
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err);

__global__ void emptyKernel() {
}

/**
 * Host function that copies the data and launches GS on the CPU
 *
 */
void  empty_kernel()
{

    struct timespec start, end;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &start);	
    emptyKernel<<<1,1>>> ();
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);	
    long long unsigned int diff = (1000000000L) * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("kernel time %llu\n",diff);
}
/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}




