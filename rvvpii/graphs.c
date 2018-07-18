#include<stdint.h>
#include<assert.h>
#include<stdlib.h>
#include<stdio.h>
#include<time.h> 
#include<string.h>
#include<limits.h>
#include "algs.h"
#define bool uint8_t
#define true 1
#define false 0
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
problem copy_problem(problem* p){
    problem p_copy;
    p_copy.n = p->n;
    p_copy.male_prefs = malloc(sizeof(int)*p->n*p->n);
    p_copy.female_prefs = malloc(sizeof(int)*p->n*p->n);
    return p_copy;
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

// returns the number of steps the algorithm takes to converge on the given problem, or -1 if it does not converge in time
int num_steps(problem* p,int* starting_match, int max_iters, int* (*alg)(problem*,int*,int*,int*)){
        int n = p->n;
        int* match = starting_match;
        int* reverse_match = reverse(n,match);
        int output_size = 0;
        int j;
        problem p_copy = copy_problem(p);
#ifdef PRINT_FAILS
        char* matches = malloc(sizeof(char)*max_iters*n*10);
        matches[0] = '\0';
        char temp[10];
        for (int  k = 0; k < n; k++){
            sprintf(temp, "\t%d", match[k]);
            strcat(matches,temp);
        }
        strcat(matches, "\n");
#endif 
        for(int i = 0; i < n*n;i++){
            p_copy.male_prefs[i] = p->male_prefs[i];
            p_copy.female_prefs[i] = p->female_prefs[i];
        }
        for (j =0; j < max_iters; j++){
            if(is_filled(p,match) && is_stable(p,match,reverse_match)){
                free(alg(&p_copy,match,reverse_match,&output_size));
                assert(output_size ==0);
                free(match);
#ifdef PRINT_FAILS
                free(matches);
#endif
                free(p_copy.male_prefs);
                free(p_copy.female_prefs);
                free(reverse_match);
                return j;
            }
            else {
                assert(output_size == 1 || j == 0);
            }
            int* output = alg(&p_copy,match,reverse_match,&output_size);
#ifdef PRINT_FAILS
            char temp[10];
            for (int  k = 0; k < n; k++){
                sprintf(temp, "\t%d", output[k]);
                strcat(matches,temp);
            }
            strcat(matches, "\n");
#endif
            free(match);
            free(reverse_match);
            match = output;
            reverse_match = reverse(n,output);
        }
#ifdef PRINT_FAILS
        printf("failed on\n");
        printf("male_prefs:\n");
        for (int j = 0; j < n; j++){
            for (int k = 0; k < n; k++){
                printf("%d ", p->male_prefs[j*n+k]);
            }
            printf("\n");
        }
        printf("female_prefs:\n");
        for (int j = 0; j < n; j++){
            for (int k = 0; k < n; k++){
                printf("%d ", p->female_prefs[j*n+k]);
            }
            printf("\n");
        }
        printf("progress:\n");
        printf("%s",matches);
        printf("\n\n");
        free(matches);
#endif
        free(p_copy.male_prefs);
        free(p_copy.female_prefs);
        free(match);
        free(reverse_match);
        return -1;
}
// returns the number of steps the algorithm takes to converge on the given problem, or -1 if it does not converge in time
// assumes correct algorithm that does not modify the problem
int num_steps_fast(problem* p,int* starting_match, int max_iters, int* (*alg)(problem*,int*,int*,int*)){
        int n = p->n;
        int* match = starting_match;
        int* reverse_match = reverse(n,match);
        int output_size;
        int j;
        for (j =0; j < max_iters; j++){
            int* output = alg(p,match,reverse_match,&output_size);
            free(match);
            free(reverse_match);
            match = output;
            reverse_match = reverse(n,output);
            if(output_size == 0){
                free(match);
                free(reverse_match);
                return j;
            }
        }
        free(match);
        free(reverse_match);
        return -1;
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
        if (num_steps(p,match,iterations,alg) != -1){
                passed++;
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
        for (int j = 0; j < n; j++) {
            match[j] = -1;
        }
        if (num_steps(p,match,iterations,alg) != -1){
                passed++;
        }
    }
    free(p->male_prefs);
    free(p->female_prefs);
    return passed/trials;
}
// average number of iterations starting from a random state or -1 if it does not converge
int avg_num_iters_rand(int n,int trials, int* (*alg)(problem*,int*,int*,int*)){
    problem pv;
    problem* p = &pv;
    p->n = n;
    p->male_prefs = malloc(sizeof(int)*n*n);
    p->female_prefs = malloc(sizeof(int)*n*n);
    
    long total_iters  = 0;
    for (int i = 0; i < trials; i++) {
        fill_random(p); 
        int* match= malloc(n*sizeof(int));
        random_match(n,match);
        int current_iters = num_steps(p,match,n*n,alg);
        if (current_iters == -1)
           return -1; 
        total_iters += current_iters;
        //printf("%d\n",current_iters);
    }
    free(p->male_prefs);
    free(p->female_prefs);
    return total_iters/trials;
}
// number of iterations starting from an all single state or -1 if it does not converge, returns [min,max,avg] over the trials
// max is INT_MAX if algorithm does not converge in time
int* num_iters_single(int n,int trials, int* (*alg)(problem*,int*,int*,int*),bool fast){
    int* output = malloc(sizeof(int)*3);
    output[0] = INT_MAX;
    output[1] = -1;
    problem pv;
    problem* p = &pv;
    p->n = n;
    p->male_prefs = malloc(sizeof(int)*n*n);
    p->female_prefs = malloc(sizeof(int)*n*n);
    
    long total_iters  = 0;
    for (int i = 0; i < trials; i++) {
        fill_random(p); 
        int* match= malloc(n*sizeof(int));
        for (int i = 0; i < n;i++)
            match[i] = -1;
        int current_iters;
        if (fast){
            current_iters = num_steps_fast(p,match,n*n,alg);
        }else{
            current_iters = num_steps(p,match,n*n,alg);
        }
        if (current_iters == -1){
           output[1] = INT_MAX; 
        }
        total_iters += current_iters;
        output[0] = current_iters < output[0] ? current_iters : output[0]; 
        output[1] = current_iters > output[1] ? current_iters : output[1]; 
        //printf("%d\n",current_iters);
    }
    free(p->male_prefs);
    free(p->female_prefs);
    output[2] = total_iters/trials;
    return output;
}
void compare_single_start(int num_algs, int* (**algs)(problem*,int*,int*,int*)){
    problem pv;
    problem* p = &pv;
    for (int n = 2; n < 20;n++) {
        p->n = n;
        p->male_prefs = malloc(sizeof(int)*n*n);
        p->female_prefs = malloc(sizeof(int)*n*n);
        int* match = malloc(sizeof(int)*n);
        int* reverse_match = malloc(sizeof(int)*n);
        for (int t = 0; t < 2; t++) {
            fill_random(p);
            for (int i =0; i < n;i++){
                match[i] = -1;
                reverse_match[i] = -1;
            }
            problem* problems = malloc(sizeof(problem)*num_algs);
            for (int i =0; i < num_algs; i++){
                problems[i] = copy_problem(p);
            }
            int** outputs = malloc(sizeof(int*)*num_algs);
            for (int i = 0; i < n*2; i++) {
               
               for (int j = 0; j < num_algs; j++) {
                   int junk;
                   outputs[j] = algs[j](p,match,reverse_match,&junk);
               }
               for (int h = 0; h < n; h++){
                   reverse_match[h] =  -1;
               }
               for (int k = 0; k < n; k++) {
                   int val = outputs[0][k];
                   for (int j = 1; j < num_algs; j++){
                       if (outputs[j][k] != val){
                           printf("different\n");
                           return;
                       }
                   }
                   match[k] = val;

                   if (val != -1)
                       reverse_match[val] = k;
               }
               for (int j = 0; j < num_algs; j++){
                   free(outputs[j]);
               }
               
           }
           for (int i = 0; i < num_algs; i++){
               free(problems[i].male_prefs);  
               free(problems[i].female_prefs);
           }
           free(outputs);
        }
        free(match);
        free(reverse_match);
        free(p->male_prefs);
        free(p->female_prefs);
    }
    
}

int main() {
    int n = 500;
    //percent = 100*convergence_rate(n,trials,iters,&pii);
    //printf("pii passed %f%% of the time\n",percent);
    int trials = 500;
    for (int i = 2; i <= n; i++){
        int* out = num_iters_single(i,trials,&pii4,true);
        printf("%d,%d\n",i,out[2]);
        if (i% 10 == 0)
            fflush(stdout);
        free(out);
    }
    /*
    int* (*algs[4])(problem*,int*,int*,int*) = {&dummy,&pii2,&pii3,&pii4};
    char* names[4] = {"pii","pii2","pii3","pii4"};
    double percent;
    int iters = n;
    */
    //compare_single_start(4,algs);
    //percent = 100*convergence_rate_all_single(n,trials,iters,&pii);
    //printf("pii passed %f%% of the time\n",percent);
    /*
    percent = 100*convergence_rate_all_single(n,trials,iters,&pii2);
    printf("pii2 passed %f%% of the time\n",percent);
    percent = 100*convergence_rate_all_single(n,trials,iters,&pii3);
    printf("pii3 passed %f%% of the time\n",percent);
    percent = 100*convergence_rate_all_single(n,trials,iters,&pii4);
    printf("pii4 passed %f%% of the time\n",percent);
    */
    /*
    int* out;
    out = num_iters_single(n,trials,&pii2);
    printf("pii2 took (%d,%d,%d) iterations \n",out[0],out[1],out[2]);
    free(out);
    out = num_iters_single(n,trials,&pii3);
    printf("pii3 took (%d,%d,%d) iterations \n",out[0],out[1],out[2]);
    free(out);
    out = num_iters_single(n,trials,&pii4);
    printf("pii4 took (%d,%d,%d) iterations \n",out[0],out[1],out[2]);
    free(out);
    */
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
