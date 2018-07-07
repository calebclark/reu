#include<stdint.h>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<string.h>
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
void swap(uint8_t* arr, int i1,int i2) {
    uint8_t swap = arr[i1];
    arr[i1] = arr[i2];
    arr[i2] = swap;
}
void printarr(int n, uint8_t* arr) {
    uint8_t prefix = 0;
    for (int i = 0; i < n; i++) {
        if (prefix)
            printf(",");
        printf("%d",arr[i]);
        prefix = 1;
    }
}
bool is_stable(problem p, uint8_t* match) {
    int n = p.n;
    for (uint8_t i = 0; i < n; i++) {
        // check if the male in the ith match pair prefers anyone to his first match, and return 0 if they prefer him
        uint8_t male = match[i];
        for (uint8_t j = 0; j < n; j++) {
            uint8_t female = p.male_prefs[male*n+j];
            // we have reached his current spouse
            if (female == i){
                break;
            }
            /* if we have made it this far we know male prefers female to his spouse we now check if female prefers male to
             * her spouse */
            if (p.female_prefs[female*n+male] < p.female_prefs[female*n+match[female]]) {
                return false;
            }

        }
    }
    return true;
}
// make sure output is filled correctly
bool is_filled(problem p, uint8_t* match) {
    int n = p.n;
    // does it contain the variables it should
    bool contains[n];
    for (int i = 0; i < n; i++)
        contains[i] = 0;
    for (int i = 0; i < n;i++) {
        if (match[i] > n || contains[match[i]])
            return false;
        contains[match[i]] = true;
    }
    return true;
}
void fill_random(problem p) {
        int n = p.n;
        // seed the random number generator
        seed();
        // fill with a random permutation
        // first fill
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p.male_prefs[i*n+j] = j;
                p.female_prefs[i*n+j] = j;
            }
        }
        // then permute using TAOCP Vol. 2 pg. 145, algorithm P
        // TODO generate random numbers better
        for (int i = 0; i < n; i++) {
            // guard need to be at the bottom since it's unsigned so we can't go below 0
            for (int j = n-1;; j--) {
                int randm = rand() % (j+1);
                int randf = rand() % (j+1);
                int swapm = p.male_prefs[i*n+randm];
                int swapf = p.female_prefs[i*n+randf];
                p.male_prefs[i*n+randm] = p.male_prefs[i*n+j];
                p.female_prefs[i*n+randf] = p.female_prefs[i*n+j];
                p.male_prefs[i*n+j] = swapm;
                p.female_prefs[i*n+j] = swapf;
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
uint8_t* generate_all_perms(uint8_t n) {
    // alloc space
    uint8_t* array = malloc(n*factorial(n)*sizeof(uint8_t));
    uint8_t* temp = malloc(n*sizeof(uint8_t));
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
int id(int n, uint8_t* perm,uint8_t* all_perms) {
    for (int i = 0; i < factorial(n); i++){
        uint8_t good = 1;
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
uint8_t* pii(problem p, uint8_t* match, uint8_t* reverse_match, int* output_size){
        int n = p.n;
        uint8_t* output = malloc(n*sizeof(uint8_t));
        for (int i = 0; i < n; i++) {
            output[i] = match[i];
        }
        int nm1[n];
        *output_size = 0;
        // find nm1 pairs 
        for (int m = 0; m < n; m++) {
            int nm1gf = -1;
            for (int f = 0; f < n;f++){
                // check for blocking and male dominance 
                if(p.male_prefs[m*n+f] < p.male_prefs[m*n+match[m]] 
                        && p.female_prefs[f*n+m] < p.female_prefs[f*n+reverse_match[f]]
                        && (nm1gf == -1 || p.male_prefs[m*n+f] < p.male_prefs[m*n+nm1gf])){
                    *output_size = 1;
                    nm1gf = f;
                }
            }
            //check for female dominance
            for (int i = 0; i < m; i++) {
                if (nm1[i] == nm1gf
                    && p.female_prefs[nm1gf*n+i] < p.female_prefs[nm1gf*n+m]) {
                    nm1gf = -1;
                } else if (nm1[i] == nm1gf) {
                    nm1[i] = -1;
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
#ifdef DEBUG
                printf("nm1 pair (%d,%d)\n", m ,nm1partner);
#endif
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
#ifdef DEBUG
                    printf("nm2 pair (%d,%d)\n", nm2row,nm2column);
#endif
                } 
            }
        }
#ifdef DEBUG
        printf("done\n");
#endif

        return output;
} 
// performs all possible divorces for this match, and returns an array of size output_size*n with the resulting permutations
uint8_t* divorce(problem p, uint8_t* match, uint8_t* reverse_match, int* output_size){
        int n = p.n;
        uint8_t* output = malloc(n*n*n*sizeof(uint8_t));
        *output_size = 0;
        // find blocking pairs
        for (int m = 0; m < n; m++) {
            for (int f = 0; f < n;f++){
                if(p.male_prefs[m*n+f] < p.male_prefs[m*n+match[m]] 
                        && p.female_prefs[f*n+m] < p.female_prefs[f*n+reverse_match[f]]){
                    for (int i = 0; i < n; i++){
                        output[*output_size*n + i] = match[i];
                    }
                    uint8_t m2 = reverse_match[f];
                    swap(output + *output_size*n,m,m2);
                    (*output_size)++;
                }
            }
        }
        return output;
} 

//TODO speed up
void printDOT(problem p, uint8_t* perms,uint8_t* (*alg)(problem,uint8_t*,uint8_t*,int*)) {
    int n = p.n;
    printf("digraph {\n");
    uint8_t match[n];
    uint8_t reverse_match[n];
    for (int i = 0; i < factorial(p.n); i++) {
        uint8_t is_stable = 1;
        for (int j = 0; j < n; j++) {
            match[j] = perms[i*n+j];
            reverse_match[perms[i*n+j]] = j;
        }
        int output_size;
        uint8_t* output = alg(p, match, reverse_match,&output_size);
        is_stable = !output_size;
        for (int j = 0; j < output_size; j++){
            printf("M%d -> M%d\n", i+1,1+id(n,output + j*n,perms));
        }
        free(output);
        if (is_stable){
            printf("M%d [style=filled,fillcolor=darkgreen,fontcolor=white]\n",i+1);
        } else {
            printf("M%d [style=filled,fillcolor=crimson,fontcolor=white]\n",i+1);

        }

    }
    printf("}\n");
}
void printpartDOT(problem p, uint8_t* perms,uint8_t* (*alg)(problem,uint8_t*,uint8_t*,int*),char* color) {
    int n = p.n;
    uint8_t match[n];
    uint8_t reverse_match[n];
    for (int i = 0; i < factorial(p.n); i++) {
        uint8_t is_stable = 1;
        for (int j = 0; j < n; j++) {
            match[j] = perms[i*n+j];
            reverse_match[perms[i*n+j]] = j;
        }
        int output_size;
        uint8_t* output = alg(p, match, reverse_match,&output_size);
        is_stable = !output_size;
        for (int j = 0; j < output_size; j++){
            //TODO get rid of magic number
            char edge[300] = "M%d -> M%d [color=";
            strcat(edge,color);
            strcat(edge,"]\n");
            printf(edge, i+1,1+id(n,output + j*n,perms));
        }
        free(output);
        if (is_stable){
            printf("M%d [style=filled,fillcolor=darkgreen,fontcolor=white]\n",i+1);
        } else {
            printf("M%d [style=filled,fillcolor=crimson,fontcolor=white]\n",i+1);

        }

    }
}

int main() {
    int n = 4;
    problem p;
    p.n = n;
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
    int tmp_male_prefs[n][n];
    int tmp_female_prefs[n][n];
    p.male_prefs = (int*)tmp_male_prefs;
    p.female_prefs = (int*)tmp_female_prefs;
    fill_random(p); 
    printf("digraph {\n");
    printpartDOT(p, perms,&divorce,"black"); 
    printpartDOT(p, perms,&pii,"orange"); 
    printf("}\n");

     
    free(perms);
}
