#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <locale.h>
#include <stdint.h>
#include <random>
#include "algs.h"
#include <assert.h>
typedef struct {
    // index into proposal array
    int proposal_index;
    // 1 if currently dating/engaged , 0 if not dating
    char is_dating;
} man_info;

/* 
 * Find a stable matching using the GS algorithm
 * returns number of iteration
 */

int GS(int n, int* male_prefs, int* female_prefs, int* output) {
    int to_return = 0;
    // 0 if an output slot has not yet been initialized
    uint8_t* output_used = (uint8_t*) calloc(sizeof(uint8_t),n);
    // flip female prefs for easy access
    // where all the men are in their proposal lists
    man_info* state = (man_info*) calloc(n,sizeof(man_info));
    // false if any man is still unmatched
    bool all_matched = false;
    bool prefix = false;
    
    for (int i = 0; i < n; i++){
       if (prefix)
           printf(",");
       bool printed = false;
       for (int j = 0; j < n; j++){ 
           if (output_used[j] && output[j] == i){
               printf("%d",j);
               printed = true;
               break;
           }
       }
       if (!printed)
           printf("-1");
        prefix = true;
    }
    printf("\n");
    while (!all_matched){
        // assume everyone is matched until we find out otherwise
        all_matched = true;
        for (int i = 0; i < n; i++) {
            if (!state[i].is_dating) {
                printf("%d is not dating\n",i);
                all_matched = false;
                int next_female = male_prefs[i*n+(state[i].proposal_index++)];
                bool swap = false;
                // propose
                if (!output_used[next_female]){
                    printf("%d proposed to %d who was single\n",i,next_female);
                    swap = true;
                }
                else if(female_prefs[next_female*n+output[next_female]] > female_prefs[next_female*n+i]) {
                   printf("%d stole %d from %d\n",i,next_female,output[next_female]);
                   state[output[next_female]].is_dating = 0;
                   swap = true;
                }
                if (swap) {
                   state[i].is_dating = 1;
                   output_used[next_female] = 1;
                   output[next_female] = i;
                   state[i].is_dating = 1;
                }
            }
        }
        bool prefix = false;
        for (int i = 0; i < n; i++){
           if (prefix)
               printf(",");
           bool printed = false;
           for (int j = 0; j < n; j++){ 
               if (output_used[j] && output[j] == i){
                   printf("%d",j);
                   printed = true;
                   break;
               }
           }
           if (!printed)
               printf("-1");
           prefix = true;
        }
        printf("\n");
        to_return++;
    }
#ifdef DEBUG
    for (int i = 0; i < n; i++) 
        assert(output_used[i]);
#endif
    free(state);
    free(output_used);
    return to_return;
}
/*
 * Builds the trivial matching (male 1 with female 1 etc...) usually not stable
 */
int trivial(int n,__attribute__((unused)) int* male_prefs,__attribute__((unused)) int* female_prefs,int* output) { 
    for (int i = 0; i < n; i++) {
        output[i] = i;
    }
    return 0;
}
