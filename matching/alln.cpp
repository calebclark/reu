#include <iostream>
#include<stdio.h>
#include<stdint.h>
#include "algs.h"
#include "tests.h"
using namespace std;
#define TRIALS  1000
#define N 2000
bool verbose = false;
void alln(int (*alg)(int,int*,int*,int*), int n, int trials);
int main() {
    alln(&GS, N, TRIALS);
}
void alln(int (*alg)(int,int*,int*,int*), int n, int trials) {
    for (int i= 2; i < n; i++) {
        printf("%d, %d\n", i, avg(i,alg,trials));
        fflush(stdout);
    }
}
