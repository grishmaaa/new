#include <stdio.h>
#include <math.h>

void hello() {
    printf("Hello, World!\n");
    // printf("The square root of 16 is %.2f\n", sqrt(16.0));
    // return 0;
}

int factorial(unsigned i ){
    if (i<=1) return 1;
    return i * factorial(i-1);
}

double calculate_e(int terms) {
    double e = 0.0;
    for (int i = 0; i < terms; i++) {
        e += 1.0 / factorial(i);
    }
    return e;
}