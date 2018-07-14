#include <stdio.h>
int factorial(int n) {
    if (n==0)
        return n;
    else
        return n*factorial(n-1);
}
int* unpack(int n, int coded) {
    int* to_return = malloc(sizeof(int)*factorial(n));
}
int main() {
    int n = 4;
    for (int i = 0; i < factorial(n); i++){

    }
}
