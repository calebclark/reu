#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <locale.h>
#include <stdint.h>
#include "algs.h"
typedef struct {
    // index into proposal array
    int proposal_index;
    // 1 if currently dating (on thread), 0 if not dating
    char is_dating;
} man_info;

/* 
 * Find a stable matching using the GS algorithm
 */
void  GS(int n, int* male_prefs, int* female_prefs, int* output) {
    // 0 if an output slot has not yet been initialized
    char* output_used = (char*) calloc(sizeof(char),n);
    // flip female prefs for easy access
    // where all the men are in their proposal lists
    man_info* state = (man_info*) calloc(n,sizeof(man_info));
    // false if any man is still unmatched
    int all_matched = 0;
    while (!all_matched){
        all_matched = 1;
        for (int i = 0; i < n; i++) {
            if (!state[i].is_dating) {
                int next_female = male_prefs[i*n+(state[i].proposal_index++)];
                all_matched = 0;
                // propose
                if (!output_used[next_female] 
                        || female_prefs[next_female*n+output[next_female]] > female_prefs[next_female*n+i]) {
                   output_used[next_female] = 1;

                   output[next_female] = i;
                   state[i].is_dating = 1;
                }
            }
        }
    }
    free(state);
    free(female_prefs);
    free(output_used);
}
