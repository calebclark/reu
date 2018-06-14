#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <locale.h>
#include <stdint.h>
int test();
char test_frame(int n, int male_prefs[n][n], int female_prefs[n][n], int correct_output[n]); 
void  GS(int n, int male_prefs[n][n], int female_prefs[n][n], int output[n]); 
void  GS2(int n, int male_prefs[n][n], int female_prefs[n][n], int output[n]); 
int main() {
    // test GS
    int test_result = test();
    if ( test_result) {
       printf("test %d failed\n",test_result);
       return 1; 
    } else {
        printf("tests pass!\n");
    }

    // seed the random number generator
    srand(time(0));
    // the number of males/females
    const int n = 40000;
    // allocate arrays
    int (*male_prefs)[n] = malloc(sizeof(int)*n*n);
    int (*female_prefs)[n] = malloc(sizeof(int)*n*n);
    int *output = malloc(sizeof(int)*n);
    if (male_prefs == NULL || female_prefs == NULL) {
        printf("malloc error\n");
        return 2;
    }
    // fill with a random permutation
    // first fill
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            male_prefs[i][j] = j;
            female_prefs[i][j] = j;
        }
    }
    // then permute using TAOCP Vol. 2 pg. 145, algorithm P
    // TODO generate random numbers better
    for (int i = 0; i < n; i++) {
        for (int j = n-1;j >= 0; j--) {
            int randm = rand() % n;
            int randf = rand() % n;
            int swapm = male_prefs[i][randm];
            int swapf = female_prefs[i][randf];
            male_prefs[i][randm] = male_prefs[i][j];
            female_prefs[i][randf] = female_prefs[i][j];
