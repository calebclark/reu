#include <iostream>
#include<stdio.h>
#include<stdint.h>
#include "algs.h"
#include "tests.h"
using namespace std;
#define TRIALS  10
#define N  1000
bool verbose = false;
void print_times(string name, long long unsigned int* times, int size); 
void test_alg(void (*alg)(int,int*,int*,int*), string name, int n, int trials); 
int main() {
    test_alg(&GS, "Sequential GS", N, TRIALS);
    test_alg(&trivial, "Trivial", N, TRIALS);


}

void test_alg(void (*alg)(int,int*,int*,int*), string name, int n, int trials) {
    int t = test_matcher(alg); 
    if (t == 0) {
       cout << name << " passed all tests\n";
       long long unsigned int* times = time_matcher(&GS, n,trials); 
       print_times(name,times,TRIALS);
    }
    else {
        cout << name << " failed test " << t;
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
    printf("average time = %'llu nanoseconds\n", sum/size );



}
