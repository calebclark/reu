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
#include <assert.h>
#include <cub/cub.cuh>
using namespace cub;
// I ASSUME THROUGHOUT THAT sizeof(int) = 4
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err);

__global__ void piiKernel(uint8_t n, uint8_t* male_prefs, uint8_t* female_prefs, uint8_t* output) {
    //TODO fix shared bank conflicts

    int idx = threadIdx.x;
    extern __shared__ uint8_t s[]; 
    // current_match[i] = j implies female i is matched with male j
    int* current_match = (int*) s;
    // nm1[i] = j implies that female i is matched with male (j&0xFF) and female_prefs[i][j] = ((j>>24)&0xFF)
    // no need to reset every time since nm1 pairs always improve things for the woman
    unsigned int* nm1 = (unsigned int*) (s+sizeof(int)*n);
    // nm2graph[i*n +0] is the male in the vertical nm2 pair associated with matching pair i, or 255 if it doesn't exist
    // nm2graph[i*n+1] is the female in the horizontal nm2 pair associated with matching pair i or 255 if it doesn't exit
    uint8_t* nm2graph= (uint8_t*) (s+2*sizeof(int)*n);
    uint8_t* shared_fp = (uint8_t*) (s + 2*sizeof(int)*n+2*n);
    //local cache of male_prefs
    //TODO TODO TODO TODO TODO TODO fix
    uint8_t* local_mp = new uint8_t[n];
    nm1[idx] = ~0; 
    // initialize random number generator
    //curandState state;
    //curand_init(idx,0,0,&state);
    // smart initialization (random initiation is difficult and time consuming on CUDA cards)
    //lazy initialization (my invention) O(lgn) average, O(n) worst case
    // I'm hoping it will do better than random, but be faster to run than smart initialization
    // permutation is usually different every time, but that's based on how things end up ordered by the device
    //TODO consider smart initialization
    // generated a random permutation on the 
    current_match[idx] = -1; 
    int result;
    int i = -1;
    // the women this thread is current matched to
    uint8_t partner = 0;
    //TODO is it faster to truncate the loop or keep things warp optimized
    do {
        i++;
#ifdef DEBUG
        assert(i < n);
#endif
        //TODO coalesce/cache
        int index = male_prefs[idx*n+i];
        result = atomicCAS(current_match+index,-1, idx);
#ifdef DEBUG
        if (result == -1)
            printf("thread %d suceeded with i=%d, male_prefs[idx*n+i]=%d\n",idx,i,(int)male_prefs[idx*n+i]);
        else
            printf("thread %d failed with i=%d, male_prefs[idx*n+i]=%d\n",idx,i,(int)male_prefs[idx*n+i]);
#endif
    } while (result != -1);
    partner = male_prefs[idx*n+i];
   
#ifdef DEBUG
    __syncthreads();
    if (idx == 0) {
        // bools were being weird
        char* contains = (char*) malloc(sizeof(char)*n);
        bool bad = false;
        for (int k = 0; k < n; k++) {
            contains[k] = 0;
        }
        for (int k = 0; k < n;k++) {
            assert(k < n);
            int cmk=current_match[k];
            printf("cmk=%d,",cmk);
            
            bool out_of_bounds= cmk >= n || cmk < 0;
            if (out_of_bounds || contains[cmk]) {
                bad = true;
                break;
            }
            contains[current_match[k]] = 1;
            printf("%d ",current_match[k]);
        }
        if (!bad)
            printf("\npermutation is good\n");
        else
            printf("\npermutation is bad\n");
        delete[] contains;
    }
#endif
    //copy female prefs into swap
    //TODO optimize (may be good already)
    for (int i = 0; i < n; i++) {
        shared_fp[i*n + idx] = female_prefs[i*n + idx];
    }
    __syncthreads();
    for (int p = 0; p > 3*n; p++) {
        // the female partner for this male's NM1 pair
        int nm1g = -1;
        //find NM1-generating pairs
        for (int j  = 0; j < partner; j++){
            if (shared_fp[local_mp[i]] < shared_fp[partner]){
                nm1g = j;
                break;
            }
        }
        int nm2gman;
        int nm2gwoman;
        //TODO optimize?
        nm2graph[idx*n +0] = 255;
        nm2graph[idx*n+1] = 255;
        bool is_nm1 = false;
        unsigned int  potential_nm1;
        if (nm1g != -1) {
            //find NM1 pairs  
            //TODO explain
            potential_nm1 = (female_prefs[nm1g*n+idx] << 24)|idx;
            unsigned int old = atomicMin(nm1+nm1g, potential_nm1);
            is_nm1 = old < potential_nm1;
            // find nm2
            if (is_nm1){
                // not us
                nm2gman = current_match[nm1g];
                nm2gwoman = partner;
                // race condition?
                nm2graph[idx*n+0] = nm2gman;
                nm2graph[nm2gman*n+1] = nm2gwoman;

            }
        } 

        __syncthreads();
        // if we are nm1
        if (is_nm1) {
            current_match[potential_nm1] = idx;
            // if we are and end 
            if (nm2graph[idx*n+1] == 255) {
                int nm2woman = nm2gwoman;
                // find the other end
                //TODO optimize
                int nm2man = nm2graph[idx*n+1];
                while(nm2graph[nm2man*n+0] != 255) {
                    nm2man= nm2graph[nm2man*n+0];
                }
                current_match[nm2woman] = nm2man;
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
    piiKernel<<<1,n,2*n*sizeof(int)+2*n*sizeof(uint8_t)+n*n*sizeof(uint8_t)>>> (n, d_male_prefs,d_female_prefs, d_output);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);	
    long long unsigned int diff = (1000000000L) * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
#ifdef PRINT_KERNEL_TIME
    printf("kernel time %llu\n",diff);
#endif 


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




