#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
    int n = 100000;
    int trials = 100000;
    int arr[n];
    srand(time(NULL));
    unsigned long long int dups = 0;
    for (int t = 0; t < trials; t++) {
        //fill array randomly
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % n;
        }
        //count duplicates 
        char contains[n];
        for (int i = 0; i < n; i++) {
            contains[i] = 0;
        }
        for (int i = 0; i < n; i++) {
            if (!contains[arr[i]])
                contains[arr[i]] = 1;
            else
                dups++;
        }
    }
    printf("avg dups=%llu\n",dups/trials);

}
