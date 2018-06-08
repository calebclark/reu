#include <stdio.h>
#include <time.h>
#include <stdlib.h>
int main() {
    const int n = 200;
    int** male_prefs = malloc(sizeof(int)*n*n);
    int** female_prefs = malloc(sizeof(int)*n*n);
    int* output= malloc(sizeof(int)*n);
    time_t start = time();
     
    time_t stop = time();
    int seconds = difftime(stop,start);
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
   int[3][3] m = { { 1, 2, 3}, {1, 2, 3},{ 1, 2, 3}};
   int[3][3] f = { { 1, 2, 3}, {1, 2, 3},{ 1, 2, 3}};
   int[3] correct = {1,2,3};
   if (!test_frame(3,m,f,correct))
       return 1;
   // test 2
   int[3][3] m = { { 3, 2, 1}, {2, 1, 3},{ 1, 3, 2}};
   int[3][3] f = { { 3, 2, 1}, {2, 1, 3},{ 1, 3, 2}};
   int[3] correct = {3,2,1};
   if (!test_frame(3,m,f,correct))
       return 1;
}
/*
 * testing framework
 * returns: 0 if correct_output is equal to the output and 0 otherwise
 */
char test_frame(int n, int** male_prefs, int** female_prefs, int** correct_output) {
    int** test_output = malloc(sizeof(int)*n);
    GS(n,male_prefs,female_prefs, test_output);
    char to_return = 0;
    for (int i = 0; i < n; i++) {
        if (test_output[i] != output[i]) {
            to _return = 1;
            break;
        }
    } 
    free(test_output);
}
/* 
 * Find a stable matching 
 * Params:
 * n - The number of males/females
 * male_prefs - An nxn array of ints representing where male_pref[i][j] = k  
 *              indicates that female k ranks jth on male i's preference list 
 * female_prefs - The female preferences with the same layout as the male preferences
 *
 * ouput - An array of length n where output is stored. It is format so output[i] = j 
 *         that male i is matched with female j.
 *
 */
void  GS(const int n, const int** male_prefs, const int** female_prefs, int* output) {
    for (int i = 0; i < n; i++) {
    }

}
