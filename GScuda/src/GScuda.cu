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
__global__ void GSKernel(int n, int* male_prefs, int* female_prefs, int* output) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
}

/**
 * Host function that copies the data and launches GS on the CPU
 *
 */
void  GS(int n, int** male_prefs, int** female_prefs, int* output)
{
	int *d_male_prefs, *d_female_prefs;
	int* d_output;

	size_t prefs_size = sizeof(int)*n*n;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_male_prefs, prefs_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_female_prefs, prefs_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_output, sizeof(int)*n));
	CUDA_CHECK_RETURN(cudaMemcpy(d_male_prefs, male_prefs, prefs_size, cudaMemcpyHostToDevice));
	
	//static const int BLOCK_SIZE = 256;
	//const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	GSKernel<<<1,n>>> (n, d_male_prefs,d_female_prefs, d_output);

	CUDA_CHECK_RETURN(cudaMemcpy(output, d_output, sizeof(int)*n, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_male_prefs));
	CUDA_CHECK_RETURN(cudaFree(d_female_prefs));
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
/*
 * Units tests
 *
 * returns: 0 if all tests pass, otherwise returns the # of the test that failed
 */
int test() {
    // test 1
   int m1[3][3] = { { 0, 1, 2}, {0, 1, 2},{ 0, 1, 2}};
   int f1[3][3] = { { 0, 1, 2}, {0, 1, 2},{ 0, 1, 2}};
   int correct1[3] = {0,1,2};
   if (test_frame(3,(int**)m1,(int**)f1,(int*)correct1))
       return 1;
   // test 2
   int m2[3][3] = { { 2, 1, 0}, {1, 0, 2},{ 0, 2, 1}};
   int f2[3][3] = { { 2, 1, 0}, {1, 0, 2},{ 0, 2, 1}};
   int correct2[3] = {2,1,0};
   if (test_frame(3,(int**)m2,(int**)f2,(int*)correct2))
       return 2;
   // test 2
   int m3[3][3] = { { 2, 0, 1}, {0, 1, 2},{ 0, 1, 2}};
   int f3[3][3] = { { 1, 0, 2}, {2, 0, 1},{ 1, 2, 0}};
   int correct3[3] = {1,2,0};
   if (test_frame(3,(int**)m3,(int**)f3,(int*)correct3))
       return 3;
   return 0;
}


int main(void)
{
    int test_result = test();
    if ( test_result) {
       printf("test %d failed\n",test_result);
       return 1;
    } else {
        printf("tests pass!");
    }

    // seed the random number generator
    srand(time(0));
    // the number of males/females
    const int n = 10000;
    // allocate arrays
    int (*male_prefs)[n] = new int[n][n];
    int (*female_prefs)[n] = new int[n][n];
    int *output = new int[n];
    if (male_prefs == NULL || female_prefs == NULL) {
        printf("malloc error\n");
        return 2;
    }
    // fill with a random permutation
    // first fill
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            male_prefs[i][j] = j;
            female_prefs[i][j] = j;
        }
    }
    // then permute using TAOCP Vol. 2 pg. 145, algorithm P
    // TODO generate random numbers better
    for (int i = 0; i < n; i++) {
        for (int j = n-1;j >= 0; j--) {
            int randm = rand() % n;
            int randf = rand() % n;
            int swapm = male_prefs[i][randm];
            int swapf = female_prefs[i][randf];
            male_prefs[i][randm] = male_prefs[i][j];
            female_prefs[i][randf] = female_prefs[i][j];
            male_prefs[i][j] = swapm;
            female_prefs[i][j] = swapf;
        }
    }


    clock_t start = clock();
    GS(n,(int**)male_prefs,(int**)female_prefs,(int*)output);
    clock_t stop = clock();
    double seconds = ((double)(stop-start))/CLOCKS_PER_SEC;
    printf("took %f seconds for n=%d\n",seconds,n);
    delete male_prefs;
    delete female_prefs;
    delete output;

	return 0;
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




