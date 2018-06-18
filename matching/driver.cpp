#include<stdio.h>
#include "algs.h"
#include "tests.h"

int main() {
    int t =test_matcher(&GS); 
    if (!t) {
       printf("GS (and the testing framework) works\n"); 
    }
    else {
       printf("failed test %d",t);
    }

    t =test_matcher(&trivial); 
    if (!t) {
        printf("trivial did not pass\n");
    } else {
        printf("trivial passed\n");
    }
    

}
