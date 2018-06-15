#include<stdio.h>
#include "algs.h"
#include "tests.h"

int main() {
    if (test_matcher(&GS)) {
       printf("GS (and the testing framework) works"); 
    }
    else {
       printf("fail");
    }

}
