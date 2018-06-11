#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
int test();
char test_frame(int n, int male_prefs[n][n], int female_prefs[n][n], int correct_output[n]); 
void  GS(int n, int male_prefs[n][n], int female_prefs[n][n], int output[n]); 
int main() {
    // test GS
    int test_result = test();
    if ( test_result) {
       printf("test %d failed\n",test_result);
       return 1; 
    }
    // seed the random number generator
    srand(time(0));
    // the number of males/females
    const int n = 60000;
    // allocate arrays
    int (*male_prefs)[n] = malloc(sizeof(int)*n*n);
    int (*female_prefs)[n] = malloc(sizeof(int)*n*n);
    int *output = malloc(sizeof(int)*n);
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
            male_prefs[i][j] = swapm;
            female_prefs[i][j] = swapf;
        }
    }


    clock_t start = clock();
    GS(n,male_prefs,female_prefs,output);
    clock_t stop = clock();
    double seconds = ((double)(stop-start))/CLOCKS_PER_SEC;
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
   int m1[3][3] = { { 0, 1, 2}, {0, 1, 2},{ 0, 1, 2}};
   int f1[3][3] = { { 0, 1, 2}, {0, 1, 2},{ 0, 1, 2}};
   int correct1[3] = {0,1,2};
   if (test_frame(3,m1,f1,correct1))
       return 1;
   // test 2
   int m2[3][3] = { { 2, 1, 0}, {1, 0, 2},{ 0, 2, 1}};
   int f2[3][3] = { { 2, 1, 0}, {1, 0, 2},{ 0, 2, 1}};
   int correct2[3] = {2,1,0};
   if (test_frame(3,m2,f2,correct2))
       return 2;
   // test 2
   int m3[3][3] = { { 2, 0, 1}, {0, 1, 2},{ 0, 1, 2}};
   int f3[3][3] = { { 1, 0, 2}, {2, 0, 1},{ 1, 2, 0}};
   int correct3[3] = {1,2,0};
   if (test_frame(3,m3,f3,correct3))
       return 3;
   return 0;
}
/*
 * testing framework
 * returns: 0 if correct_output is equal to the output and 1 otherwise
 */
char test_frame(int n, int male_prefs[n][n], int female_prefs[n][n], int correct_output[n]) {
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
// a struct holding state information for men
typedef struct {
    // index into proposal array
    int proposal_index;
    // 1 if currently dating (on thread), 0 if not dating
    char is_dating;
} man_info;



/* 
 * Find a stable matching 
 * Params:
 * n - The number of males/females
 * male_prefs - An nxn array of ints representing where male_pref[i][j] = k  
 *              indicates that female k ranks jth on male i's preference list 
 *
 * female_prefs - An nxn array of ints representing where female_pref[i][j] = k  
 *              indicates that male k ranks jth on female i's preference list.  
 * ouput - An array of length n where output is stored. It is format so output[i] = j 
 *         that female i is matched with male j.
 *
 */
void  GS(int n, int male_prefs[n][n], int female_prefs[n][n], int output[n]) {
    // 0 if an output slot has not yet been initialized
    char* output_used = calloc(sizeof(char),n);
    // flip female prefs for easy access
    int (*fast_female)[n] = malloc(sizeof(int)*n*n);
    for (int f = 0; f < n; f++) {
       for (int r = 0; r < n; r++) {
            int m = female_prefs[f][r];
            fast_female[f][m] = r;
       }
    } 
    // where all the men are in their proposal lists
    man_info* state = calloc(n,sizeof(man_info));
   
    // false if any man is still unmatched
    int all_matched = 0;
    while (!all_matched){
        all_matched = 1;
        for (int i = 0; i < n; i++) {
            if (!state[i].is_dating) {
                int next_female = male_prefs[i][state[i].proposal_index++];
                all_matched = 0;
                // propose
                if (!output_used[next_female] 
                        || fast_female[next_female][output[next_female]] > fast_female[next_female][i]) {
                   output_used[next_female] = 1;

                   output[next_female] = i;
                   state[i].is_dating = 1;
                }
            }
        }
    }
    free(state);
    free(fast_female);
    free(output_used);
}

                        



