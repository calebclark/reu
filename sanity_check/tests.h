#include <stdint.h>
#ifndef TESTS
#define TESTS
/* 
 * Tests a stable matching algorithm against a battery of test cases. Return 0 if it passed and 1 otherwise
 * alg - An algorithm that takes:
 *          - A number of males/females
 *          - An nxn row major array where arr[i][j] = k indicates that male i ranks female k jth.
 *          - An nxn row major array where arr[i][j] = k indicates that female i ranks male k jth.
 *          - An array of size n where output will be stored such that arr[i]=j indicates that female i is paired with male j.
 *       
 */
int test_matcher(int (*alg)(int,int*,int*,int*));
/**
 * Times a stable matching algorithm on random data.
 * alg - an algorithm, with the same specification as the one described in test
 * n - the number of males/females to test with
 * t - the number of trials to run
 * returns an array of size t with the runtime of each trial in nano seconds 
 */
long long unsigned int* time_matcher(int (*alg)(int,int*,int*,int*),int n, int t);
int* freqs_matcher(int (*alg)(int,int*,int*,int*),int n, int t, int* max);


#endif
