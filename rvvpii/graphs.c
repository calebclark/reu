#include<stdint.h>
#include<assert.h>
#include<stdlib.h>
#include<stdio.h>
#include<time.h> 
#include<string.h>
#include<limits.h>
#define bool uint8_t
#define true 1
#define false 0
typedef struct {
    int n;
    int* male_prefs;
    int* female_prefs;
} problem;
__attribute__((const)) int factorial(uint8_t n) {
    if (n==0)
        return 1;
    return n*factorial(n-1);
}
uint8_t seeded = 0;
void seed(){
    if (!seeded){
        srand(time(0));
    }
    seeded=1;
}
void swap(int* arr, int i1,int i2) {
    int swap = arr[i1];
    arr[i1] = arr[i2];
    arr[i2] = swap;
}
void printarr(int n, int* arr) {
    uint8_t prefix = 0;
    for (int i = 0; i < n; i++) {
        if (prefix)
            printf(",");
        printf("%d",arr[i]);
        prefix = 1;
    }
}
void printarr4(int n, int* arr) {
    bool prefix = false;
    for (int i = 0; i < n; i++) {
        if (prefix)
            printf(",");
        printf("%d",arr[i]);
        prefix = true;
    }
}
bool is_stable(problem* p, int* match,int* reverse_match) {
    int n = p->n;
    for (int m = 0; m < n; m++) {
        for (int f = 0; f < n; f++) {
            if (p->male_prefs[m*n+f] < p->male_prefs[m*n+match[m]]
                    &&  p->female_prefs[f*n+m] < p->female_prefs[f*n+reverse_match[f]])
                return false;
        }
    }
    return true;
}
// make sure output is filled correctly
bool is_filled(problem* p, int* match) {
    int n = p->n;
    // does it contain the variables it should
    bool contains[n];
    for (int i = 0; i < n; i++)
        contains[i] = 0;
    for (int i = 0; i < n;i++) {
        if (match[i] > n || match[i] < 0 || contains[match[i]])
            return false;
        contains[match[i]] = true;
    }
    return true;
}
void fill_random(problem* p) {
        int n = p->n;
        // seed the random number generator
        seed();
        // fill with a random permutation
        // first fill
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p->male_prefs[i*n+j] = j;
                p->female_prefs[i*n+j] = j;
            }
        }
        // then permute using TAOCP Vol. 2 pg. 145, algorithm P
        // TODO generate random numbers better
        for (int i = 0; i < n; i++) {
            // guard need to be at the bottom since it's unsigned so we can't go below 0
            for (int j = n-1;; j--) {
                int randm = rand() % (j+1);
                int randf = rand() % (j+1);
                int swapm = p->male_prefs[i*n+randm];
                int swapf = p->female_prefs[i*n+randf];
                p->male_prefs[i*n+randm] = p->male_prefs[i*n+j];
                p->female_prefs[i*n+randf] = p->female_prefs[i*n+j];
                p->male_prefs[i*n+j] = swapm;
                p->female_prefs[i*n+j] = swapf;
                if (j == 0) 
                    break;
            }
        }
}
/*
 * given an integer n retruns an array of size n*(n!) containing all permutations of size n.
 *
 * from TAOCP vol 4
 * 
 */ 
int* generate_all_perms(int n) {
    // alloc space
    int* array = malloc(n*factorial(n)*sizeof(int));
    int* temp = malloc(n*sizeof(int));
    // initialize the first permutation
    for (int i = 0; i < n; i++){
        temp[0*n+i] = i;
    }
    // TAOCP, vol 4, algorithm L
    for (int i = 0; i < factorial(n); i++){
        for (int o = 0; o < n; o++) {
            array[i*n+o] = temp[o];
        }
        int j = n-2;
        while (j >= 0 && temp[j] >= temp[j+1]) {
            j--;
        }
        if (j < 0){
            free(temp);
            return array;
        }
        int l = n-1;
        while (temp[j] >= temp[l]) {
            l--;
        }
        swap(temp,j,l);
        for (int k = j+1, l2 = n-1; k < l2;k++,l2--){
            swap(temp,k,l2);
        }
    }
    return NULL;
}
//TODO speed up
int id(int n, int* perm,int* all_perms) {
    for (int i = 0; i < factorial(n); i++){
        int good = 1;
        for (int j = 0; j < n; j++){
            if (all_perms[i*n+j] != perm[j]){
                good = 0;
            }
        }
        if (good) {
            return i;
        }
    }
    return -1;
}
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

        return output;
} 
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
// performs all possible divorces for this match, and returns an array of size output_size*n with the resulting permutations
int* divorce(problem* p, int* match, int* reverse_match, int* output_size){
        int n = p->n;
        int* output = malloc(n*n*n*sizeof(int));
        *output_size = 0;
        // find blocking pairs
        for (int m = 0; m < n; m++) {
            for (int f = 0; f < n;f++){
                if(p->male_prefs[m*n+f] < p->male_prefs[m*n+match[m]] 
                        && p->female_prefs[f*n+m] < p->female_prefs[f*n+reverse_match[f]]){
                    for (int i = 0; i < n; i++){
                        output[*output_size*n + i] = match[i];
                    }
                    int m2 = reverse_match[f];
                    swap(output + *output_size*n,m,m2);
                    (*output_size)++;
                }
            }
        }
        return output;
} 

void random_match(int n, int* match){
    for (int k = 0;k < n; k++) {
        int j= rand() % (k+1);
        match[k] = match[j];
        match[j] = k;
    }
}
int* reverse(int n, int* match){
    int* to_return = malloc(n*sizeof(int));
    for (int i = 0; i < n; i++){
        to_return[i] = -1;
    }
    for (int i = 0; i < n; i++){
        if (match[i] != -1)
            to_return[match[i]] = i;
    }
    return to_return;
}
double convergence_rate(int n, int trials, int iterations, int* (*alg)(problem*,int*,int*,int*)){
    problem pv;
    problem* p = &pv;
    p->n = n;
    p->male_prefs = malloc(sizeof(int)*n*n);
    p->female_prefs = malloc(sizeof(int)*n*n);
    
    double passed = 0;
    for (int i = 0; i < trials; i++) {
        fill_random(p); 
        int* match= malloc(n*sizeof(int));
        random_match(n,match);
        int* reverse_match = reverse(n,match);
        int output_size;
        for (int j =0; j < iterations; j++){
            int* output = alg(p,match,reverse_match,&output_size);
            assert(output_size == 1||output_size == 0);
            free(match);
            free(reverse_match);
            match = output;
            reverse_match = reverse(n,output);
        }
        if(is_filled(p,match) && is_stable(p,match,reverse_match)){
            passed++;
        }
        free(reverse_match);
        free(match);
        if (i % 100 == 0){
            fprintf(stderr,"PROGRESS: finished %d/%d trials\n",i,trials);
        }
    }
    free(p->male_prefs);
    free(p->female_prefs);
    return passed/trials;
}
double convergence_rate_all_single(int n, int trials, int iterations, int* (*alg)(problem*,int*,int*,int*)){
    problem pv;
    problem* p = &pv;
    p->n = n;
    p->male_prefs = malloc(sizeof(int)*n*n);
    p->female_prefs = malloc(sizeof(int)*n*n);
    
    double passed = 0;
    for (int i = 0; i < trials; i++) {
        fill_random(p); 
        int* match= malloc(n*sizeof(int));
        for (int i = 0; i < n; i++) {
            match[i] = -1;
        }
        int* reverse_match = reverse(n,match);
        int output_size;
        for (int j =0; j < iterations; j++){
            int* output = alg(p,match,reverse_match,&output_size);
            assert(output_size == 1||output_size == 0);
            free(match);
            free(reverse_match);
            match = output;
            reverse_match = reverse(n,output);
        }
        if(is_filled(p,match) && is_stable(p,match,reverse_match)){
            passed++;
        }
        free(reverse_match);
        free(match);
        if (i % 100 == 0){
            fprintf(stderr,"PROGRESS: finished %d/%d trials\n",i,trials);
        }
    }
    free(p->male_prefs);
    free(p->female_prefs);
    return passed/trials;
}
int main() {
    int n = 100;
    int iters = n;
    int trials = 10000;
    double percent;
    //percent = 100*convergence_rate_all_single(n,trials,iters,&pii);
    //printf("pii passed %f%% of the time\n",percent);
    //percent = 100*convergence_rate_all_single(n,trials,iters,&pii2);
    //printf("pii2 passed %f%% of the time\n",percent);
    percent = 100*convergence_rate_all_single(n,trials,iters,&pii3);
    printf("pii3 passed %f%% of the time\n",percent);
    //percent = 100*convergence_rate_all_single(n,trials,iters,&pii4);
    //printf("pii4 passed %f%% of the time\n",percent);
    /*
    int t = 300;
    int n = 4;
    uint8_t* perms = generate_all_perms(n);
#ifdef DEBUG
    for (int i = 0; i < factorial(n); i++) {
        printf("%d: ",i+1);
        printarr(n,perms+i*n);
        printf("\n");
    }
#endif
        
    // from "transformation from arbitrary matchings to stable matchings", Tamura, 1990
    // male_prefs[i][j] is the rating male i give female j
    //int tmp_male_prefs[4][4] = {{0,2,1,3},{3,0,2,1},{1,3,0,2},{2,1,3,0}};
    //int tmp_female_prefs[4][4] = {{2,0,3,1},{1,2,0,3},{3,1,2,0},{0,3,1,2}};
    //printf("digraph {\n");
    //printpartDOT(p, perms,&divorce,"black"); 
    //printpartDOT(p, perms,&pii,"orange"); 
    //printf("}\n");
    for (int i = 0; i < t; i++) {
        fill_random(p); 
        bool* g1 = get_graph(p, perms,&pii); 
        bool* g2 = get_graph(p, perms,&divorce); 
        tc(factorial(n),g1);
        tc(factorial(n),g2);
        if (!is_subgraph(factorial(n),g1,g2)) {
            printf("male_prefs:\n");
            for (int j = 0; j < n; j++){
                printarr4(n,p->male_prefs+j*n);
                printf("\n");
            }
            printf("female_prefs:\n");
            for (int j = 0; j < n; j++){
                printarr4(n,p->male_prefs+j*n);
                printf("\n");
            }
        }

    }

     
    free(perms);
    */
}
