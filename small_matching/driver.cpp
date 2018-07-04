#include <iostream>
#include<stdio.h>
#include<stdint.h>
#include "algs.h"
#include "tests.h"
#include "pii.h"
#include "cuda_test.h"
using namespace std;
#define TRIALS 1 
#define N 150 
bool verbose = false;
void print_times(string name, long long unsigned int* times, int size); 
void test_alg(void (*alg)(uint8_t,uint8_t*,uint8_t*,uint8_t*), string name, uint8_t n, int trials); 
void time_alg(void (*alg)(uint8_t,uint8_t*,uint8_t*,uint8_t*), string name, uint8_t n, int trials); 
void run_alg(void (*alg)(uint8_t,uint8_t*,uint8_t*,uint8_t*), string name, uint8_t n, int trials); 
void convergence_rate_printer(void (*alg)(uint8_t,uint8_t*,uint8_t*,uint8_t*), string name, uint8_t n, int trials);
int main() {
//    for (int i = 0; i < 10000; i++)
 //       empty_kernel();
    //convergence_rate_printer(&GS,"GS",100,100000);
 //   convergence_rate_printer(&trivial,"Trival",4,10000);
    //convergence_rate_printer(&pii,"pii",100,100000);
    //int num_tests = 0;
    //int t = test_matcher_loose(&pii,&num_tests);
//    printf("failed %d/%d tests\n",t,num_tests);
    time_alg(&pii, "pii", 100, 1);
//    time_alg(&GS, "GS", 10, 1);
}
void convergence_rate_printer(void (*alg)(uint8_t,uint8_t*,uint8_t*,uint8_t*), string name, uint8_t n, int trials){
    double rate = convergence_rate(alg,n,trials);
    cout << name << " has a convergence rate of " << rate << " with n=" << (int)n << endl;
}
void time_alg(void (*alg)(uint8_t,uint8_t*,uint8_t*,uint8_t*), string name, uint8_t n, int trials) {
       long long unsigned int* times = time_matcher(alg, n,trials); 
       print_times(name,times,trials);
       free(times);
}

void run_alg(void (*alg)(uint8_t,uint8_t*,uint8_t*,uint8_t*), string name, uint8_t n, int trials) {
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
    long long unsigned int mask = 1;
    mask <<=sizeof(long long unsigned int)*8-1;
    setlocale(LC_NUMERIC, "");
    cout << "times for: " << name << endl;
    long long unsigned int sum = 0;
    for (int i = 0; i < size;i++){
        if (verbose) {    
            printf("elapsed time = %'llu nanoseconds\n",  times[i]);
        }
        if (sum & times[i] & mask){
            fprintf(stderr,"OVERFLOW in print_times, exiting\n");
        }
        sum += times[i];

    }
    printf("average time = %'llu nanoseconds\n", sum/size );



}
