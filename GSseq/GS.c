#include <stdio.h>
#include <time.h>
#include <stdlib.h>
int test();
char test_frame(int n, int** male_prefs, int** female_prefs, int* correct_output); 
void  GS(int n, int** male_prefs, int** female_prefs, int* output); 
int main() {
    if (test()) {
       printf("tests fail\n");
       return 1; 
    }
    const int n = 200;
    int** male_prefs = malloc(sizeof(int)*n*n);
    int** female_prefs = malloc(sizeof(int)*n*n);
    int* output= malloc(sizeof(int)*n);
    time_t start = time(NULL);
     
    time_t stop = time(NULL);
    double seconds = difftime(stop,start);
    printf("took %f seconds for n=%d",seconds,n); 
    free(male_prefs);
    free(female_prefs);
    free(output);
}
/*
 * Units tests 
 *
 * returns: 0 if all tests pass, otherwise returns the # of the test that failed
 */
int test() {
    // test 1
   int m1[3][3] = { { 1, 2, 3}, {1, 2, 3},{ 1, 2, 3}};
   int f1[3][3] = { { 1, 2, 3}, {1, 2, 3},{ 1, 2, 3}};
   int correct1[3] = {1,2,3};
   if (!test_frame(3,(int**)m1,(int**)f1,correct1))
       return 1;
   // test 2
   int m2[3][3] = { { 3, 2, 1}, {2, 1, 3},{ 1, 3, 2}};
   int f2[3][3] = { { 3, 2, 1}, {2, 1, 3},{ 1, 3, 2}};
   int correct2[3] = {3,2,1};
   if (!test_frame(3,(int**)m2,(int**)f2,correct2))
       return 2;
   return 0;
}
/*
 * testing framework
 * returns: 0 if correct_output is equal to the output and 0 otherwise
 */
char test_frame(int n, int** male_prefs, int** female_prefs, int* correct_output) {
    int* test_output = malloc(sizeof(int)*n);
    GS(n,male_prefs,female_prefs, test_output);
    char to_return = 0;
    for (int i = 0; i < n; i++) {
        if (test_output[i] != correct_output[i]) {
            to_return = 1;
            break;
        }
    } 
    free(test_output);
    return to_return;
}
/* 
 * Find a stable matching 
 * Params:
 * n - The number of males/females
 * male_prefs - An nxn array of ints representing where male_pref[i][j] = k  
 *              indicates that female k ranks jth on male i's preference list 
 *
 * ouput - An array of length n where output is stored. It is format so output[i] = j 
 *         that male i is matched with female j.
 *
 */
void  GS(int n, int** male_prefs, int** female_prefs, int* output) {
    // false if any man is still unmatched
    all_matched = 0;
    while (!all_matched){
        for (int i = 0

    }
}
