#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <locale.h>
#include <stdint.h>
int test();
char test_frame(int n, int* male_prefs, int* female_prefs, int* correct_output); 
void fill_random(int n, int* male_prefs, int* female_prefs) {
        // fill with a random permutation
        // first fill
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                male_prefs[i*n+j] = j;
                female_prefs[i*n+j] = j;
            }
        }
        // then permute using TAOCP Vol. 2 pg. 145, algorithm P
        // TODO generate random numbers better
        for (int i = 0; i < n; i++) {
            for (int j = n-1;j >= 0; j--) {
                int randm = rand() % n;
                int randf = rand() % n;
                int swapm = male_prefs[i*n+randm];
                int swapf = female_prefs[i*n+randf];
                male_prefs[i*n+randm] = male_prefs[i*n+j];
                female_prefs[i*n+randf] = female_prefs[i*n+j];
                male_prefs[i*n+j] = swapm;
                female_prefs[i*n+j] = swapf;
            }
        }
}
uint64_t* time((*alg)(int,int[][],int,int*),int n, int t);
    // seed the random number generator
    srand(time(0));
    uint64_t times = malloc(sizeof(uint64_t)*t);
    for (int trial = 0; trial < t; trial++) {
        // the number of males/females
        const int n = 20000;
        // allocate arrays
        int* male_prefs = (int*) malloc(sizeof(int)*n*n);
        int* female_prefs = (int*)malloc(sizeof(int)*n*n);
        int* output = (int*) malloc(sizeof(int)*n);
        if (male_prefs == NULL || female_prefs == NULL) {
            printf("malloc error\n");
            return 2;
        }
        fill_random(n, male_prefs, female_prefs);
        // from https://www.cs.rutgers.edu/~pxk/416/notes/c-tutorials/gettime.html
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);	
        alg(n,male_prefs,female_prefs,output);
        clock_gettime(CLOCK_MONOTONIC, &end);	
        uint64_t diff = (1000000000L) * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
        setlocale(LC_NUMERIC, "");
        printf("elapsed time = %'llu nanoseconds\n", (long long unsigned int) diff);
        free(male_prefs);
        free(female_prefs);
        free(output);
        times[t] = diff;
    }
    return times;
    
}
/*
 * Unit tests 
 *
 * returns: 0 if all tests pass, otherwise returns the # of the test that failed
 */
int test() {
    // test 1
   int m1[3][3] = { { 0, 1, 2}, {0, 1, 2},{ 0, 1, 2}};
   int f1[3][3] = { { 0, 1, 2}, {0, 1, 2},{ 0, 1, 2}};
   int correct1[3] = {0,1,2};
   if (test_frame(3,(int*)m1,(int*)f1,(int*)correct1))
       return 1;
   // test 2
   int m2[3][3] = { { 2, 1, 0}, {1, 0, 2},{ 0, 2, 1}};
   int f2[3][3] = { { 2, 1, 0}, {1, 0, 2},{ 0, 2, 1}};
   int correct2[3] = {2,1,0};
   if (test_frame(3,(int*)m2,(int*)f2,(int*)correct2))
       return 2;
   // test 2
   int m3[3][3] = { { 2, 0, 1}, {0, 1, 2},{ 0, 1, 2}};
   int f3[3][3] = { { 1, 0, 2}, {2, 0, 1},{ 1, 2, 0}};
   int correct3[3] = {1,2,0};
   if (test_frame(3,(int*)m3,(int*)f3,(int*)correct3))
       return 3;
   return 0;
}
