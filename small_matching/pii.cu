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

__global__ void piiKernel(uint8_t n, uint8_t* male_prefs, uint8_t* female_prefs, uint8_t* output) {
    //TODO fix shared bank conflicts

    int idx = threadIdx.x;
    extern __shared__ int s[]; 
    // current_match[i] = j implies female i is matched with male j
    int* current_match = (int*) s;
    //curent_match flipped for fast lookups so match current[j] = i implies female i is matched with male j
    int* match_current = (int*) &current_match[n];
    // nm1[i] = j implies that female i is matched with male (j&0xFF) and female_prefs[i][j] = ((j>>24)&0xFF)
    // no need to reset every time since nm1 pairs always improve things for the woman
    unsigned int* nm1 = (unsigned int*) &match_current[n];
    // an array of three tuples of the form nm2graph[male id(row)] = (row neighbor index, female id (column),column neighbor index), indexes are 255 if null
    // where the row (respectively column) neighbor index is the index into nm2graph of the the nm2 pair it is connected to by the matching pair in the same row as it
    uint8_t* nm2graph= (uint8_t*) &nm1[n];
    uint8_t* shared_fp = (uint8_t*) &nm2graph[3*n];
    uint8_t* unstable = (uint8_t*) &shared_fp[n*n];
    //local cache of male_prefs
    //TODO coalesce
    uint8_t* local_mp = new uint8_t[n];
    for (int i = 0; i < n; i++){
        local_mp[i] = male_prefs[idx*n+i];
    }
    // generated a random permutation on the CPU
    int temp_man = current_match[idx] = output[idx]; 
    match_current[temp_man] = idx;
    //copy female prefs into swap
    //TODO optimize (may be good already)
    for (int i = 0; i < n; i++) {
        shared_fp[i*n + idx] = female_prefs[i*n + idx];
    }
    __syncthreads();
    for (int p = 0; p < n; p++) {
        __syncthreads();
        int partner = match_current[idx]; 
        // the female partner for this male's NM1 pair
        int nm1g = -1;
        //TODO optimize
        unstable[0] = 0;
        __syncthreads();
        //find NM1-generating pairs
        for (int j  = 0; local_mp[j] != partner; j++){
            if (shared_fp[local_mp[j]*n+idx] < shared_fp[local_mp[j]*n+current_match[local_mp[j]]]){
                nm1g = local_mp[j];
#ifdef DEBUG
                printf("found nm1g pair for male %d\n",idx);
#endif
                unstable[0] = 1;
                break;
            }
        }
#ifdef DEBUG
        printf("thread: %d, potential nm1g=%d\n",idx,nm1g);
#endif
        //TODO optimize
        __syncthreads();
        if (unstable[0] == 0){
#ifdef DEBUG
        printf("FOUND STABLE MATCHING\n");
#endif
            break;
        }
        //TODO optimize?
        nm2graph[idx*3 +0] = 255;
        nm2graph[idx*3 +1] = 255;
        nm2graph[idx*3+2] = 255;
        bool is_nm1 = false;
        unsigned int  potential_nm1_construct = ~0;
        // initialize large
        nm1[idx] = INT_MAX;
        __syncthreads();
        if (nm1g != -1) {
            //find NM1 pairs  
            //TODO explain
            potential_nm1_construct = (female_prefs[nm1g*n+idx] << 24)|idx;
            atomicMin(nm1+nm1g, potential_nm1_construct);
        } 
        __syncthreads();
        is_nm1 = (potential_nm1_construct == nm1[nm1g]);
        // find nm2
        if (is_nm1){
                // race condition?
                // horizontal nm2 pair 
                nm2graph[idx*3+0] = current_match[nm1g];
                //vertical nm2 pair
                nm2graph[(current_match[nm1g])*3+1] = partner;
                nm2graph[(current_match[nm1g])*3+2] = idx;

            }
#ifdef DEBUG
                printf("thread: %d ready to sync\n",idx);
#endif

        __syncthreads();
        // if we are nm1
        if (is_nm1) {
#ifdef DEBUG
                printf("thread: %d, final nm1g=%d\n",idx,nm1g);
#endif
            current_match[nm1g] = idx;
            match_current[idx] = nm1g;
            // if we are a column end 
            if (nm2graph[idx*3+2] == 255) {
#ifdef DEBUG
                printf("thread: %d, nm2graph[idx] = (%d,%d,%d)\n",idx,nm2graph[idx*3+0],nm2graph[idx*3+1],nm2graph[idx*3+2]);

#endif
                int nm2column = partner;
                //TODO optimize
                int nm2row= current_match[nm1g];
#ifdef DEBUG
                printf("thread: %d, nm2row=%d, nm2graph[nm2row] = (%d,%d,%d)\n",idx,nm2row,nm2graph[nm2row*3+0],nm2graph[nm2row*3+1],nm2graph[nm2row*3+2]);
#endif
                // if it's a single pair
                if (nm2graph[nm2row*3+0]==nm2graph[nm2row*3+2]){
                    nm2column = nm2graph[nm2row*3+1];
                }
                // if it's a chain
                else {
                    while(nm2graph[nm2row*3+0] != 255) {
#ifdef DEBUG
                        printf("thread: %d, nm2row=%d, nm2graph[nm2row] = (%d,%d,%d)\n",idx,nm2row,nm2graph[nm2row*3+0],nm2graph[nm2row*3+1],nm2graph[nm2row*3+2]);
#endif

                        nm2row = nm2graph[nm2row*3+0];
                    }
                }
#ifdef DEBUG
                printf("final nm2row=%d,nm2column=%d\n",nm2row,nm2column);
#endif
                current_match[nm2column] = nm2row;
                match_current[nm2row] = nm2column;

            }
                    



        }
    }
    // copy output out
    output[idx] = current_match[idx];
    delete[] local_mp;
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
    //knuth shuffle from sgb
    for (uint8_t k = 0;k < n; k++) {
        uint8_t j= rand() % (k+1);
        output[k] = output[j];
        output[j] = k;
    }
#ifdef DEBUG
    printf("male_prefs: \n ");
    for (int i = 0; i < n; i++){ 
        for (int j = 0; j < n; j++)
            printf("%hhd ",male_prefs[i*n+j]);
        printf("\n ");
    }
    printf("female_prefs: \n ");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%hhd ",female_prefs[i*n+j]);
        printf("\n ");
    }
    printf("permutation: ");
    for (int i = 0; i < n; i++) 
        printf("%hhd ",output[i]);
    printf("\n ");
#endif
	CUDA_CHECK_RETURN(cudaMemcpy(d_output, output, n*sizeof(uint8_t), cudaMemcpyHostToDevice));
    piiKernel<<<1,n,n*sizeof(int)+n*sizeof(int)+n*sizeof(unsigned int) + 3*n*sizeof(uint8_t) + n*n*sizeof(uint8_t)+1>>> (n, d_male_prefs,d_female_prefs, d_output);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);	
    long long unsigned int diff = (1000000000L) * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    //printf("kernel time %llu\n",diff);
	CUDA_CHECK_RETURN(cudaMemcpy(output, d_output, sizeof(uint8_t)*n, cudaMemcpyDeviceToHost));
#ifdef DEBUG
    printf("result: ");
    for (int i = 0; i < n; i++) 
        printf("%hhd ",output[i]);
    printf("\n ");
#endif
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




