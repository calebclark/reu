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
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>
#include <cub/cub.cuh>
using namespace cub;

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err);

__global__ void piiKernel(uint8_t n, uint8_t* male_prefs, uint8_t* female_prefs, uint8_t* output) {
    int idx = threadIdx.x;
    extern __shared__ int current_match[]; 
    // initialize random number generator
    curandState state;
    curand_init(idx,0,0,&state);

}


#ifdef DEBUG
    if (idx == 0) {
        bool* contains = new bool[n];
        bool bad = false;
        for (int i = 0; i < n; i++)
            contains[i] = 0;
        for (int i = 0; i < n;i++) {
            if (current_match[i] > n || contains[current_match[i]])
                bad = true;
            contains[current_match[i]] = true;
            printf("%d ",current_match[i]);
        }
        if (!bad)
            printf("\npermutation is good\n");
        else
            printf("\npermutation is bad\n");
        delete[] contains;
    }
#endif




}

/**
 * Host function that copies the data and launches GS on the CPU
 *
 */
void  pii(uint8_t n, uint8_t* male_prefs, uint8_t* female_prefs, uint8_t* output)
{
	uint8_t *d_male_prefs, *d_female_prefs;
	uint8_t* d_output;

	size_t prefs_size = sizeof(uint8_t)*n*n;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_male_prefs, prefs_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_female_prefs, prefs_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_output, sizeof(uint8_t)*n));
	CUDA_CHECK_RETURN(cudaMemcpy(d_male_prefs, male_prefs, prefs_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_female_prefs, female_prefs, prefs_size, cudaMemcpyHostToDevice));


    struct timespec start, end;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &start);	
    piiKernel<<<1,n,n*sizeof(int)>>> (n, d_male_prefs,d_female_prefs, d_output);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);	
    long long unsigned int diff = (1000000000L) * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

    printf("kernel time %llu\n",diff);


	CUDA_CHECK_RETURN(cudaMemcpy(output, d_output, sizeof(uint8_t)*n, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_male_prefs));
	CUDA_CHECK_RETURN(cudaFree(d_female_prefs));
	CUDA_CHECK_RETURN(cudaFree(d_output));
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




