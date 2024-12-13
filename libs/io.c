#include <stdio.h>

int read() {
    int x;
    scanf("%d", &x);
    return x;
}

void write(int x) {
    printf("%d\n", x);
}