#include <iostream>
#include<stdio.h>
#include<stdint.h>
#include "algs.h"
#include "tests.h"
using namespace std;
#define TRIALS  100000
#define N  100
bool verbose = false;
void rounds_alg(int (*alg)(int,int*,int*,int*), int n, int trials); 
int main() {
    int std_male_prefs_reversed[5][5] = {{3,2,4,1,0},{3,2,4,0,1},{0,2,4,3,1},{1,2,0,4,3},{2,1,3,0,4}};
    int std_female_prefs[5][5] = {{2,1,3,0,4},{3,2,1,0,4},{4,2,3,0,1},{3,1,2,4,0},{3,2,4,0,1}};
    int* std_male_prefs = (int*) malloc(25*sizeof(int));
    int n = 5;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n;j++) {
            int rank = std_male_prefs_reversed[i][j];
            std_male_prefs[i*n+rank] = j;
        }
    }
    int output[5];
    GS(5,std_male_prefs,(int*)&std_female_prefs,(int*)&output);
    free(std_male_prefs);
    //rounds_alg(&GS, N, TRIALS);
}
void rounds_alg(int (*alg)(int,int*,int*,int*), int n, int trials) {
       int max;
       int* freqs = freqs_matcher(alg, n,trials, &max); 
       for (int i = 0; i <= max; i++){
           printf("%d, %d\n",i, freqs[i]);
       } 
       free(freqs);
}

