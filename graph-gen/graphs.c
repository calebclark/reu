#include<stdint.h>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
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
void fill_random(int n, int* male_prefs, int* female_prefs) {
        // seed the random number generator
        seed();
        // fill with a random permutation
        // first fill
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                male_prefs[i*n+j] = j;
                female_prefs[i*n+j] = j;
            }
        }
        // then permute using TAOCP Vol. 2 pg. 145, algorithm P
        // TODO generate random numbers better
        for (int i = 0; i < n; i++) {
            // guard need to be at the bottom since it's unsigned so we can't go below 0
            for (int j = n-1;; j--) {
                int randm = rand() % (j+1);
                int randf = rand() % (j+1);
                int swapm = male_prefs[i*n+randm];
                int swapf = female_prefs[i*n+randf];
                male_prefs[i*n+randm] = male_prefs[i*n+j];
                female_prefs[i*n+randf] = female_prefs[i*n+j];
                male_prefs[i*n+j] = swapm;
                female_prefs[i*n+j] = swapf;
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
}
int id(int n, uint8_t* perm,uint8_t* all_perms) {
    for (int i = 0; i < factorial(n); i++){
        uint8_t good = 1;
        for (int j = 0; j < n; j++){
            if (all_perms[i*n+j] != perm[j]){
                good = 0;
            }
        }
#ifdef DEBUG
        printf("comparing ");
        printarr(n,perm);
        printf(" with ");
        printarr(n,all_perms+i*n);
#endif
        if (good) {
#ifdef DEBUG
            printf("\nmatch\n");
#endif
            return i;
        }
#ifdef DEBUG
            printf("\nno match\n");
#endif
    }
    return -1;
}
// performs the kth divorce for this matching in place, outputs 1 if a divorce was performed and 0 otherwise
int divorce(int n, uint8_t* match,int* male_prefs,int* female_prefs){
} 

//TODO speed up
void printDivDOT(int n, uint8_t* perms,int* male_prefs,int* female_prefs) {
    printf("digraph {\n");
    uint8_t match[n];
    uint8_t reverse_match[n];
    for (int i = 0; i < factorial(n); i++) {
        uint8_t is_stable = 1;
        for (int j = 0; j < n; j++) {
            match[j] = perms[i*n+j];
            reverse_match[perms[i*n+j]] = j;
        }
#ifdef DEBUG
        printf("working with permutation: ");
        printarr(n,match);
        printf("\n");
#endif
        // find blocking pairs
        for (int m = 0; m < n; m++) {
            for (int f = 0; f < n;f++){
#ifdef DEBUG
                printf("pair m=%d,f=%d,match[m]=%d,reverse_match[f]=%d\n",m,f,match[m],reverse_match[f]);
                printf("%d < %d && %d  < %d \n",male_prefs[m*n+f],male_prefs[m*n+match[m]],female_prefs[f*n+m],female_prefs[f*n+reverse_match[f]]);
#endif
                if(male_prefs[m*n+f] < male_prefs[m*n+match[m]] 
                        && female_prefs[f*n+m] < female_prefs[f*n+reverse_match[f]]){
                    is_stable = 0;
#ifdef DEBUG
                    printf("blocking pair m=%d,f=%d\n",m,f);
#endif
                    uint8_t temp[n];
                    for (int i = 0; i < n; i++){
                        temp[i] = match[i];
                    }
                    uint8_t m2 = reverse_match[f];
                    swap(temp,m,m2);
#ifdef DEBUG
        printarr(n,match);
        printf(" -> ");
        printarr(n,temp);
        printf("\n");
#endif
                    printf("M%d -> M%d\n", i+1,1+id(n,temp,perms));

                }
            }
        }
        if (is_stable){
            printf("M%d [style=filled,fillcolor=darkgreen,fontcolor=white]\n",i+1);
        } else {
            printf("M%d [style=filled,fillcolor=crimson,fontcolor=white]\n",i+1);

        }

    }

    printf("}\n");
}

int main() {
    const int n = 5;
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
    //int male_prefs[4][4] = {{0,2,1,3},{3,0,2,1},{1,3,0,2},{2,1,3,0}};
    //int female_prefs[4][4] = {{2,0,3,1},{1,2,0,3},{3,1,2,0},{0,3,1,2}};
    int male_prefs[n][n];
    int female_prefs[n][n];
    fill_random(n, (int*) male_prefs, (int*)female_prefs); 
    printDivDOT(n, perms,(int*) male_prefs, (int*)female_prefs); 

     
    free(perms);
}
