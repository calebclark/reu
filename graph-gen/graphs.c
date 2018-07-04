/**
 *
 * given an array of all permutations of integer 0,1, ..., n-2, returns an array of all permutations of integers
 * 0,1,...,n-1
 * TAKEN FROM SMALL_MATCHING
 */ 
int * generate_next_perm(int n, int nfactorial, int nfactorial_old, int* all_perms_old) {
#ifdef FALSE//DEBUG
       printf("Permutations for n=%d, nfactorial=%d, nfactorial_old=%d\n",n,nfactorial,nfactorial_old);
#endif
       int* all_perms = (int*)malloc(sizeof(int)*nfactorial*n);
       if (all_perms==NULL) {
           fprintf(stderr, "Malloc error in generate_next_perm: n=%d, nfactorial=%d\n",n,nfactorial);
           exit(1);
       }
       for (uint8_t i = 0; i < nfactorial_old; i++) {
           for (uint8_t j = 0; j < n; j++){
               // copy all the number over, skipping spot j
               for (uint8_t k_old = 0,k_new = 0; k_new < n; k_old++,k_new++) {
                   if (k_new==j){
                       k_new++;
                       all_perms[i*n*n + j*n+j] = n-1;
                       if (k_new == n)
                           break;
                   }
                   all_perms[i*n*n+j*n+k_new] = all_perms_old[i*(n-1)+k_old];
               }

#ifdef FALSE //DEBUG
               for (int k = 0; k < n; k++) {
                   printf("%d ",all_perms[i*n*n+j*n+k]); 
               }
               printf("\n");
#endif
           }
       }
       return all_perms;

}

/*
 * given an integer n retruns an array of size n*(n!) containing all permutations of size n.
 *
 * obviously there are faster ways, but I've already written this one, and I don't think speed is of essence right now.
 * 
 */ 
int* generate_all_perms(int n) {
    int first = calloc(1,1);
    int nfac
    for (int i = 1; i <= n; i++) {


    }
}

int main() {

}
