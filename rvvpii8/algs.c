#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <limits.h>

#include "algs.h"
// performs all possible PII's for this match, and returns an array of size output_size*n with the resulting permutations
int* pii(problem* p, int* match, int* reverse_match, int* output_size){
        int n = p->n;
        int* output = malloc(n*sizeof(int));
        for (int i = 0; i < n; i++) {
            output[i] = match[i];
        }
        int nm1[n];
        *output_size = 0;
        // find nm1 pairs
        for (int m = 0; m < n; m++) {
            int nm1gf = -1;
            nm1[m] = -1;
            for (int f = 0; f < n;f++){
                // check for blocking and male dominance
                if(p->male_prefs[m*n+f] < p->male_prefs[m*n+match[m]]
                        && p->female_prefs[f*n+m] < p->female_prefs[f*n+reverse_match[f]]
                        && (nm1gf == -1 || p->male_prefs[m*n+f] < p->male_prefs[m*n+nm1gf])){
                    *output_size = 1;
                    nm1gf = f;
                }
            }
            if (nm1gf != -1) {
                //check for female dominance
                for (int i = 0; i < m; i++) {
                    if (nm1[i] == nm1gf
                        && p->female_prefs[nm1gf*n+i] < p->female_prefs[nm1gf*n+m]) {
                        nm1gf = -1;
                        break;
                    } else if (nm1[i] == nm1gf) {
                        nm1[i] = -1;
                    }

                }
            }
            nm1[m] = nm1gf;
        }
        // quit if it's stable
        if (!*output_size)
            return output;
        // find nm2g pairs
        // nm2g[male id(row)] (row neighbor index, female id(column), column neighbor index)
        int nm2g[n][3];
        for (int i = 0; i < n; i++){
            nm2g[i][0] = -1;
            nm2g[i][2] = -1;
        }
        for (int m = 0; m < n; m++) {
            if (nm1[m] != -1) {
                int partner = match[m];
                int nm1partner = nm1[m];
                output[m] = nm1partner;
                nm2g[m][0] = reverse_match[nm1partner];
                nm2g[reverse_match[nm1partner]][1] = partner;
                nm2g[reverse_match[nm1partner]][2] = m;
            }
        }
        for (int m = 0; m < n; m++) {
            if (nm1[m] != -1) {
                if (nm2g[m][2] == -1) {
                    int nm2column = match[m];
                    int nm2row = reverse_match[nm1[m]];
                    if (nm2g[nm2row][0]!=nm2g[nm2row][2]){
                        while (nm2g[nm2row][0] != -1) {
                            nm2row = nm2g[nm2row][0];
                        }
                    }
                    output[nm2row] = nm2column;
                }
            }
        }

        return output;
}
int* dummy(problem* p, int* match,__attribute__((unused)) int* reverse_match, int* output_size){
        *output_size = 1;
        int n = p->n;
        int* output = malloc(n*sizeof(int));
        for (int i = 0; i < n; i++)
            output[i] = match[i];
        return output;

}
// performs all possible PII's for this match, and returns an array of size output_size*n with the resulting permutations
int* pii2(problem* p, int* match, int* reverse_match, int* output_size){
        int n = p->n;
        int* output = malloc(n*sizeof(int));
        for (int i = 0; i < n; i++) {
            output[i] = match[i];
        }
        int nm1[n];
        *output_size = 0;
        // find nm1 pairs
        for (int m = 0; m < n; m++) {
            int nm1gf = -1;
            nm1[m] = -1;
            for (int f = 0; f < n;f++){
                // check for blocking and male dominance
                if(     (match[m] == -1 ||p->male_prefs[m*n+f] < p->male_prefs[m*n+match[m]])
                        && (reverse_match[f]==-1 || p->female_prefs[f*n+m] < p->female_prefs[f*n+reverse_match[f]])
                        && (nm1gf == -1 || p->male_prefs[m*n+f] < p->male_prefs[m*n+nm1gf])){
                    *output_size = 1;
                    nm1gf = f;
                }
            }
            if (nm1gf != -1) {
                //check for female dominance
                for (int i = 0; i < m; i++) {
                    if (nm1[i] == nm1gf
                        && p->female_prefs[nm1gf*n+i] < p->female_prefs[nm1gf*n+m]) {
                        nm1gf = -1;
                        break;
                    } else if (nm1[i] == nm1gf) {
                        nm1[i] = -1;
                    }

                }
            }
            nm1[m] = nm1gf;
        }
        // quit if it's stable
        if (!*output_size)
            return output;
        for (int m = 0; m < n; m++) {
            if (nm1[m] != -1) {
                int nm1partner = nm1[m];
                output[m] = nm1partner;
                if (reverse_match[nm1partner] != -1 && nm1[reverse_match[nm1partner]] == -1){
                    output[reverse_match[nm1partner]] = -1;
                }
            }
        }

        return output;
}
int* pii3(problem* p, int* match, int* reverse_match, int* output_size){
        int n = p->n;
        int* output = malloc(n*sizeof(int));
        for (int i = 0; i < n; i++) {
            output[i] = match[i];
        }
        int nm1[n];
        *output_size = 0;
        // find nm1 pairs
        for (int m = 0; m < n; m++) {
            int nm1gf = -1;
            nm1[m] = -1;
            for (int f = 0; f < n;f++){
                // check for blocking and male dominance
                if(     (match[m] == -1 ||p->male_prefs[m*n+f] < p->male_prefs[m*n+match[m]])
                        && (reverse_match[f]==-1 || p->female_prefs[f*n+m] < p->female_prefs[f*n+reverse_match[f]])
                        && (nm1gf == -1 || p->male_prefs[m*n+f] < p->male_prefs[m*n+nm1gf])){
                    *output_size = 1;
                    nm1gf = f;
                }
            }
            if (nm1gf != -1) {
                //check for female dominance
                for (int i = 0; i < m; i++) {
                    if (nm1[i] == nm1gf
                        && p->female_prefs[nm1gf*n+i] < p->female_prefs[nm1gf*n+m]) {
                        nm1gf = -1;
                        break;
                    } else if (nm1[i] == nm1gf) {
                        nm1[i] = -1;
                    }

                }
            }
            nm1[m] = nm1gf;
        }
        // quit if it's stable
        if (!*output_size)
            return output;
        for (int m = 0; m < n; m++) {
            if (nm1[m] != -1) {
                int nm1partner = nm1[m];
                output[m] = nm1partner;
                if (match[m] != -1){
                    p->male_prefs[m*n+match[m]] = INT_MAX;
                    p->female_prefs[match[m]*n+m] = INT_MAX;
                }
                if (reverse_match[nm1partner] != -1 && nm1[reverse_match[nm1partner]] == -1){
                    output[reverse_match[nm1partner]] = -1;
                }
            }
        }
        assert(output!=NULL);
        return output;
}
// only singles males act
int* pii4(problem* p, int* match, int* reverse_match, int* output_size){
        int n = p->n;
        int* output = malloc(n*sizeof(int));
        for (int i = 0; i < n; i++) {
            output[i] = match[i];
        }
        int nm1[n];
        *output_size = 0;
        // find nm1 pairs
        for (int m = 0; m < n; m++) {
            int nm1gf = -1;
            nm1[m] = -1;
            for (int f = 0; f < n;f++){
                // check for blocking and male dominance
                if(     (match[m] == -1 )
                        && (reverse_match[f]==-1 || (p->female_prefs[f*n+m] < p->female_prefs[f*n+reverse_match[f]]))
                        && (nm1gf == -1 || p->male_prefs[m*n+f] < p->male_prefs[m*n+nm1gf])){
                    *output_size = 1;
                    nm1gf = f;
                }
            }
            if (nm1gf != -1) {
                //check for female dominance
                for (int i = 0; i < m; i++) {
                    if (nm1[i] == nm1gf
                        && p->female_prefs[nm1gf*n+i] < p->female_prefs[nm1gf*n+m]) {
                        nm1gf = -1;
                        break;
                    } else if (nm1[i] == nm1gf) {
                        nm1[i] = -1;
                    }

                }
            }
            nm1[m] = nm1gf;
        }
        // quit if it's stable
        if (!*output_size)
            return output;
        for (int m = 0; m < n; m++) {
            if (nm1[m] != -1) {
                int nm1partner = nm1[m];
                output[m] = nm1partner;
                if (reverse_match[nm1partner] != -1 && nm1[reverse_match[nm1partner]] == -1){
                    output[reverse_match[nm1partner]] = -1;
                }
            }
        }

        return output;
}

