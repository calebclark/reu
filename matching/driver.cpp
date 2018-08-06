#include <iostream>
#include<stdio.h>
#include<stdint.h>
#include "algs.h"
#include "tests.h"
using namespace std;
#define TRIALS  10000000
#define N  100
bool verbose = false;
void prop_alg(int (*alg)(int,int*,int*,int*), int n, int trials);
int main() {
    prop_alg(&GS, N, TRIALS);
}
void prop_alg(int (*alg)(int,int*,int*,int*), int n, int trials) {
    int max;
    int* scores = freq_dist(n,alg,trials, &max);
    for (int i = 0; i <= max; i++) {
        printf("%d, %d\n", i, scores[i]);
    }
    free(scores);
}
