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
    uint8_t proposal_index;
    // 1 if currently dating/engaged , 0 if not dating
    char is_dating;
} man_info;

/* 
 * Find a stable matching using the GS algorithm
 */
void  GS(uint8_t n, uint8_t* male_prefs, uint8_t* female_prefs, uint8_t* output) {
    // 0 if an output slot has not yet been initialized
    uint8_t* output_used = (uint8_t*) calloc(sizeof(uint8_t),n);
    // flip female prefs for easy access
    // where all the men are in their proposal lists
    man_info* state = (man_info*) calloc(n,sizeof(man_info));
    // false if any man is still unmatched
    bool all_matched = false;
    while (!all_matched){
        // assume everyone is matched until we find out otherwise
        all_matched = true;
        for (uint8_t i = 0; i < n; i++) {
            if (!state[i].is_dating) {
                all_matched = false;
                uint8_t next_female = male_prefs[i*n+(state[i].proposal_index++)];
                bool swap = false;
                // propose
                if (!output_used[next_female]){
                    swap = true;
                }
                else if(female_prefs[next_female*n+output[next_female]] > female_prefs[next_female*n+i]) {
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
    }
#ifdef DEBUG
    for (uint8_t i = 0; i < n; i++) 
        assert(output_used[i]);
#endif
    free(state);
    free(output_used);
}
/* 
 * Find a stable matching using the GS algorithm, but with a queue
 */
void  GSqueue(uint8_t n, uint8_t* male_prefs, uint8_t* female_prefs, uint8_t* output) {
    // 0 if an output slot has not yet been initialized
    uint8_t* output_used = (uint8_t*) calloc(sizeof(uint8_t),n);
    // flip female prefs for easy access
    // where all the men are in their proposal lists
    man_info* state = (man_info*) calloc(n,sizeof(man_info));
    // false if any man is still unmatched
    bool all_matched = false;
    while (!all_matched){
        // assume everyone is matched until we find out otherwise
        all_matched = true;
        for (uint8_t i = 0; i < n; i++) {
            if (!state[i].is_dating) {
                all_matched = false;
                uint8_t next_female = male_prefs[i*n+(state[i].proposal_index++)];
                bool swap = false;
                // propose
                if (!output_used[next_female]){
                    swap = true;
                }
                else if(female_prefs[next_female*n+output[next_female]] > female_prefs[next_female*n+i]) {
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
    }
#ifdef DEBUG
    for (uint8_t i = 0; i < n; i++) 
        assert(output_used[i]);
#endif
    free(state);
    free(output_used);
}
/*
 * Builds the trivial matching (male 1 with female 1 etc...) usually not stable
 */
void trivial(uint8_t n,__attribute__((unused)) uint8_t* male_prefs,__attribute__((unused)) uint8_t* female_prefs,uint8_t* output) { 
    for (uint8_t i = 0; i < n; i++) {
        output[i] = i;
    }
}
