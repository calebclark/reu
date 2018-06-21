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
}
void fill_random(uint8_t n, uint8_t* male_prefs, uint8_t* female_prefs) {
        // seed the random number generator
        seed();
        // fill with a random permutation
        // first fill
        for (uint8_t i = 0; i < n; i++) {
            for (uint8_t j = 0; j < n; j++) {
                male_prefs[i*n+j] = j;
                female_prefs[i*n+j] = j;
            }
        }
        // then permute using TAOCP Vol. 2 pg. 145, algorithm P
        // TODO generate random numbers better
        for (uint8_t i = 0; i < n; i++) {
            // guard need to be at the bottom since it's unsigned so we can't go below 0
            for (uint8_t j = n-1;; j--) {
                uint8_t randm = rand() % n;
                uint8_t randf = rand() % n;
                uint8_t swapm = male_prefs[i*n+randm];
                uint8_t swapf = female_prefs[i*n+randf];
                male_prefs[i*n+randm] = male_prefs[i*n+j];
                female_prefs[i*n+randf] = female_prefs[i*n+j];
                male_prefs[i*n+j] = swapm;
                female_prefs[i*n+j] = swapf;
                if (j == 0) 
                    break;
            }
        }
}
long long unsigned int* time_matcher(void (*alg)(uint8_t,uint8_t*,uint8_t*,uint8_t*),uint8_t n, int t){
    long long unsigned int* times = (long long unsigned int*) calloc(sizeof(long long unsigned int),t);
    for (int trial = 0; trial < t; trial++) {
        // allocate arrays
        uint8_t* male_prefs = (uint8_t*) malloc(sizeof(uint8_t)*n*n);
        uint8_t* female_prefs = (uint8_t*)malloc(sizeof(uint8_t)*n*n);
        uint8_t* output = (uint8_t*) malloc(sizeof(uint8_t)*n);
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
bool is_stable(uint8_t n, uint8_t* male_prefs, uint8_t* female_prefs, uint8_t* match) {

    
    for (uint8_t i = 0; i < n; i++) {
        // check if the male in the ith match pair prefers anyone to his first match, and return 0 if they prefer him
        uint8_t male = match[i];
        for (uint8_t j = 0; j < n; j++) {
            uint8_t female = male_prefs[male*n+j];
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
// make sure output is filled correctly
bool is_filled(uint8_t n, uint8_t* match) {
    // does it contain the variables it should
    bool contains[n];
    for (int i = 0; i < n; i++)
        contains[i] = 0;
    for (int i = 0; i < n;i++) {
        if (match[i] > n || contains[match[i]])
            return false;
        contains[match[i]] = true;
    }
    return true;
}
/**
 * given an array of all permutations of integer 0,1, ..., n-2, returns an array of all permutations of integers
 * 0,1,...,n-1
 */ 
uint8_t* generate_next_perm(uint8_t n, uint8_t nfactorial, uint8_t nfactorial_old, uint8_t* all_perms_old) {
#ifdef FALSE//DEBUG
       printf("Permutations for n=%d, nfactorial=%d, nfactorial_old=%d\n",n,nfactorial,nfactorial_old);
#endif
       uint8_t* all_perms = (uint8_t*)malloc(sizeof(uint8_t)*nfactorial*n);
       if (all_perms==NULL) {
           fprintf(stderr, "Malloc error in generate_next_perm: n=%d, nfactorial=%d\n",n,nfactorial);
           exit(1);
       }
       for (uint8_t i = 0; i < nfactorial_old; i++) {
           for (uint8_t j = 0; j < n; j++){
               // copy all the number over, skipping spot j
               for (uint8_t k_old = 0,k_new = 0; k_new < n; k_old++,k_new++) {
                   if (k_new==j){
                       k_new++;
                       all_perms[i*n*n + j*n+j] = n-1;
                       if (k_new == n)
                           break;
                   }
                   all_perms[i*n*n+j*n+k_new] = all_perms_old[i*(n-1)+k_old];
               }

#ifdef FALSE //DEBUG
               for (uint8_t k = 0; k < n; k++) {
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
int test_matcher(void (*alg)(uint8_t,uint8_t*,uint8_t*,uint8_t*)) {
   int test_num = 0;
   // tests all possible preferences list for matrices up to size TEST_MAX
   uint8_t nfactorial = 1;
   uint8_t nfactorial_old = 0;
   // all permutations of 0..n-1
   uint8_t* all_perms;
   // all permutations of 0..n-2
   uint8_t* all_perms_old = (uint8_t*) malloc(sizeof(uint8_t));
   all_perms_old[0] = 0;
   for (uint8_t n = 1; n <= TEST_MAX; n++) {
#ifdef DEBUG
       printf("starting tests with n=%d\n",n);
#endif
       // update factorial calculations
       nfactorial_old = nfactorial;
       nfactorial = n*nfactorial;
       all_perms = generate_next_perm(n,nfactorial,nfactorial_old,all_perms_old);
       free(all_perms_old);
       // set up lists 
       uint8_t* prefs= (uint8_t*) malloc(sizeof(uint8_t)*n*n*2);
       uint8_t* male_prefs = prefs;
       uint8_t* female_prefs = prefs + n*n;
       uint8_t* output = (uint8_t*) malloc(sizeof(uint8_t)*n);
       // the prefence permutation each male/female is organized males 0..n-1 then females 0..n-1
       uint8_t* perm= (uint8_t*) malloc(sizeof(uint8_t)*n*2);
       //initialize 
       for (uint8_t i = 0; i < n*2;i++) {
           perm[i] = 0;
           memcpy(prefs+i*n, all_perms,sizeof(uint8_t)*n);
       }
       // test on all possible preferences
       bool done = false;
       while (!done) {
           test_num++;
#ifdef DEBUG
           printf("running test %d\n",test_num);
#endif
           alg(n,male_prefs,female_prefs,output);
           if (!is_filled(n,output) || !is_stable(n,male_prefs,female_prefs,output)) {
               return test_num;
           }
           // move to the next permutation
           uint8_t curr = 0;
           perm[curr]++;
           while (perm[curr] == nfactorial) {
               perm[curr] = 0;
               memcpy(prefs+curr*n, all_perms+perm[curr]*n,sizeof(uint8_t)*n);
               curr++;
               if (curr == 2*n) {
                   done = true;
                   break;
               }
               perm[curr]++;
           }
           if (!done)
               memcpy(prefs+curr*n, all_perms+perm[curr]*n,sizeof(uint8_t)*n);
       }

       all_perms_old = all_perms; 
       free(prefs);
       free(output);
       free(perm);
   }
   free(all_perms);  
   return 0;
}
