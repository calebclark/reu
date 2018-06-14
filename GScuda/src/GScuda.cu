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

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err);
/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void GSKernel(int n, int* male_prefs, int* fast_female, int* output) {
	//find a female to propose to
	male_prefs[thread]
}

__global__ void FFKernel (int n, int* female_prefs, int* fast_female) {
	int second_index = female_prefs[blockIdx.x*n+threadIdx.x];
	fast_female[blockIdx.x*n+second_index] = threadIdx.x;
}

/**
 * Host function that copies the data and launches GS on the CPU
 *
 */
void  GS(int n, int** male_prefs, int** female_prefs, int* output)
{
	int *d_male_prefs, *d_female_prefs,*d_fast_female;
	char *d_is_engaged;
	int *d_next_female;
	int* d_output;

	size_t prefs_size = sizeof(int)*n*n;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_male_prefs, prefs_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_female_prefs, prefs_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_fast_female, prefs_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_output, sizeof(int)*n));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_is_engaged, sizeof(char)*n));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_next_female, sizeof(char)*n));
	CUDA_CHECK_RETURN(cudaMemcpy(d_male_prefs, male_prefs, prefs_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_female_prefs, female_prefs, prefs_size, cudaMemcpyHostToDevice));


	FFKernel<<<n,n>>>  (n,d_female_prefs,d_fast_female);

	//static const int BLOCK_SIZE = 256;
	//const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	// make fast_female


	GSKernel<<<1,n>>> (n, d_male_prefs,d_female_prefs, d_output);

	CUDA_CHECK_RETURN(cudaMemcpy(output, d_output, sizeof(int)*n, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_male_prefs));
	CUDA_CHECK_RETURN(cudaFree(d_female_prefs));
	CUDA_CHECK_RETURN(cudaFree(d_fast_female));
	CUDA_CHECK_RETURN(cudaFree(d_output));
}



/*
 * testing framework
 * returns: 0 if correct_output is equal to the output and 1 otherwise
 */
char test_frame(int n, int** male_prefs, int** female_prefs, int* correct_output) {
    int* test_output = (int*) malloc(sizeof(int)*n);
    GS(n,male_prefs,female_prefs, test_output);
    char to_return = 0;
    for (int i = 0; i < n; i++) {
        if (test_output[i] != correct_output[i]) {
            to_return = 1;
            break;
        }
    }
    free(test_output);
    return to_return;
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




