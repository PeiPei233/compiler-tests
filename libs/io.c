#include <stdio.h>

int read() {
    char ch;
    int f = 1, res;
    while (ch = getchar_unlocked(), (ch < '0' || ch > '9') && ch != '-');
    if (ch == '-') {
        f = -1;
        ch = getchar_unlocked();
    }
    res = (ch ^ '0');
    while (ch = getchar_unlocked(), ch >= '0' && ch <= '9') {
        res = (res << 3) + (res << 1) + (ch ^ '0');
    }
    return res * f;
}

char __buf[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '\n', 0};

void write(int x) {
    if (x == 0) {
        putchar_unlocked('0');
        putchar_unlocked('\n');
        return;
    }
    if (x < 0) {
        putchar_unlocked('-');
        x = -x;
    }
    char *p = __buf + 18;
    while (x) {
        *--p = x % 10 + '0';
        x /= 10;
    }
    while (p < __buf + 19) {
        putchar_unlocked(*p++);
    }
}