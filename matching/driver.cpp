#include <iostream>
#include<stdio.h>
#include<stdint.h>
#include "algs.h"
#include "tests.h"
using namespace std;
#define TRIALS  1000000
#define N  10
bool verbose = false;
void print_times(string name, long long unsigned int* times, int size); 
void test_alg(void (*alg)(int,int*,int*,int*), string name, int n, int trials); 
void time_alg(void (*alg)(int,int*,int*,int*), string name, int n, int trials); 
int main() {
    time_alg(&GS, "Sequential GS", N, TRIALS);
    time_alg(&trivial, "Trivial", N, TRIALS);
}
void time_alg(void (*alg)(int,int*,int*,int*), string name, int n, int trials) {
       long long unsigned int* times = time_matcher(alg, n,trials); 
       print_times(name,times,TRIALS);
       free(times);
}
void run_alg(void (*alg)(int,int*,int*,int*), string name, int n, int trials) {
    int t = test_matcher(alg); 
    if (t == 0) {
       cout << name << " passed all tests\n";
       long long unsigned int* times = time_matcher(alg, n,trials); 
       print_times(name,times,TRIALS);
       free(times);
    }
    else {
       cout << name << " failed test " << t << endl;
    }
}
void print_times(string name, long long unsigned int* times, int size) {
    setlocale(LC_NUMERIC, "");
    cout << "times for: " << name << endl;
    long long unsigned int sum = 0;
    for (int i = 0; i < size;i++){
        if (verbose) {    
            printf("elapsed time = %'llu nanoseconds\n",  times[i]);
        }
        sum += times[i];
    }
    printf("average time = %'llu nanoseconds\n", sum/size);
}
