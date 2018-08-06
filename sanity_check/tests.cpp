#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <locale.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "tests.h"
#include <random>
#define TEST_MAX 3
int test();
// 0 if the random number generator has not been seeded, 1 otherwise
char seeded = 0;
void seed(){
    if (!seeded){
        srand(time(0));
    }
    seeded= 1;
}
void fill_random(int n, int* male_prefs, int* female_prefs) {
        // seed the random number generator
        seed();
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
long long unsigned int* time_matcher(void (*alg)(int,int*,int*,int*),int n, int t){
    long long unsigned int* times = (long long unsigned int*) calloc(sizeof(long long unsigned int),t);
    for (int trial = 0; trial < t; trial++) {
        // allocate arrays
        int* male_prefs = (int*) malloc(sizeof(int)*n*n);
        int* female_prefs = (int*)malloc(sizeof(int)*n*n);
        int* output = (int*) malloc(sizeof(int)*n);
        if (male_prefs == NULL || female_prefs == NULL) {
            printf("malloc error\n");
            return times;
        }
        fill_random(n, male_prefs, female_prefs);
        // from https://www.cs.rutgers.edu/~pxk/416/notes/c-tutorials/gettime.html
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);	
        alg(n,male_prefs,female_prefs,output);
        clock_gettime(CLOCK_MONOTONIC, &end);	
        long long unsigned int diff = (1000000000L) * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
        times[trial] = diff;
        free(male_prefs);
        free(female_prefs);
        free(output);
    }
    return times;
    
}
/**
 * n - the number of males/females
 * NOTE that the following two are not the same 
 * male_prefs - the male preference list, where male_prefs[i][j] = k indicates that male i ranks female k jth
 * female_prefs - the analogous female preference list where female_prefs[i][j] = k indicates that female i ranks male j kth 
 * match - a mathing where match[i] = j represents a that female i is paired with male j
 * returns true if the match is stable and false otherwise 
 */
bool is_stable(int n, int* male_prefs, int* female_prefs, int* match) {

    
    for (int i = 0; i < n; i++) {
        // check if the male in the ith match pair prefers anyone to his first match, and return 0 if they prefer him
        int male = match[i];
        for (int j = 0; j < n; j++) {
            int female = male_prefs[male*n+j];
            // we have reached his current spouse
            if (female == i){
                break;
            }
            /* if we have made it this far we know male prefers female to his spouse we now check if female prefers male to
             * her spouse */
            if (female_prefs[female*n+male] < female_prefs[female*n+match[female]]) {
                return false;
            }

        }
    }
    return true;
}
/**
 * given an array of all permutations of integer 0,1, ..., n-2, returns an array of all permutations of integers
 * 0,1,...,n-1
 */ 
int* generate_next_perm(int n, int nfactorial, int nfactorial_old, int* all_perms_old) {
#ifdef FALSE//DEBUG
       printf("Permutations for n=%d, nfactorial=%d, nfactorial_old=%d\n",n,nfactorial,nfactorial_old);
#endif
       int* all_perms = (int*)malloc(sizeof(int)*nfactorial*n);
       if (all_perms==NULL) {
           fprintf(stderr, "Malloc error in generate_next_perm: n=%d, nfactorial=%d\n",n,nfactorial);
           exit(1);
       }
       for (int i = 0; i < nfactorial_old; i++) {
           for (int j = 0; j < n; j++){
               // copy all the number over, skipping spot j
               for (int k_old = 0,k_new = 0; k_new < n; k_old++,k_new++) {
                   if (k_new==j){
                       k_new++;
                       all_perms[i*n*n + j*n+j] = n-1;
                       if (k_new == n)
                           break;
                   }
                   all_perms[i*n*n+j*n+k_new] = all_perms_old[i*(n-1)+k_old];
               }

#ifdef FALSE //DEBUG
               for (int k = 0; k < n; k++) {
                   printf("%d ",all_perms[i*n*n+j*n+k]); 
               }
               printf("\n");
#endif
           }
       }
       return all_perms;

}
/*
 * Unit tests 
 *
 * returns: 0 if all tests pass, and the number of the test that failed otherwise 
 */
int test_matcher(void (*alg)(int,int*,int*,int*)) {
   int test_num = 0;
   // tests all possible preferences list for matrices up to size TEST_MAX
   int nfactorial = 1;
   int nfactorial_old = 0;
   // all permutations of 0..n-1
   int* all_perms;
   // all permutations of 0..n-2
   int* all_perms_old = (int*) malloc(sizeof(int));
   all_perms_old[0] = 0;
   for (int n = 1; n <= TEST_MAX; n++) {
#ifdef DEBUG
       printf("starting tests with n=%d\n",n);
#endif
       // update factorial calculations
       nfactorial_old = nfactorial;
       nfactorial = n*nfactorial;
       all_perms = generate_next_perm(n,nfactorial,nfactorial_old,all_perms_old);
       free(all_perms_old);
       // set up lists 
       int* prefs= (int*) malloc(sizeof(int)*n*n*2);
       int* male_prefs = prefs;
       int* female_prefs = prefs + n*n;
       int* output = (int*) malloc(sizeof(int)*n);
       // the prefence permutation each male/female is organized males 0..n-1 then females 0..n-1
       int* perm= (int*) malloc(sizeof(int)*n*2);
       //initialize 
       for (int i = 0; i < n*2;i++) {
           perm[i] = 0;
           memcpy(prefs+i*n, all_perms,sizeof(int)*n);
       }
       // test on all possible preferences
       bool done = false;
       while (!done) {
           test_num++;
#ifdef DEBUG
           printf("running test %d\n",test_num);
#endif
           alg(n,male_prefs,female_prefs,output);
           if (!is_stable(n,male_prefs,female_prefs,output)) {
               return test_num;
           }
           // move to the next permutation
           int curr = 0;
           perm[curr]++;
           while (perm[curr] == nfactorial) {
               perm[curr] = 0;
               memcpy(prefs+curr*n, all_perms+perm[curr]*n,sizeof(int)*n);
               curr++;
               if (curr == 2*n) {
                   done = true;
                   break;
               }
               perm[curr]++;
           }
           if (!done)
               memcpy(prefs+curr*n, all_perms+perm[curr]*n,sizeof(int)*n);
       }

       all_perms_old = all_perms; 
       free(prefs);
       free(output);
       free(perm);
   }
   free(all_perms);  
   return 0;
}

int* freqs_matcher(int (*alg)(int,int*,int*,int*),int n, int t, int* max){
    int* to_return = (int*) calloc(sizeof(int),n*n);
    *max = 0;
    for (int trial = 0; trial < t; trial++) {
        // allocate arrays
        int* male_prefs = (int*) malloc(sizeof(int)*n*n);
        int* female_prefs = (int*)malloc(sizeof(int)*n*n);
        int* output = (int*) malloc(sizeof(int)*n);
        fill_random(n, male_prefs, female_prefs);
        int result = alg(n,male_prefs,female_prefs,output);
        to_return[result]++;
        *max = *max < result ? result : *max;
        free(male_prefs);
        free(female_prefs);
        free(output);
    }
    return to_return;
}
