#include <stdio.h>
#include <stdint.h>

#ifdef WRAP_MAIN

int __real_main(int argc, char **argv);

#else   // no WRAP_MAIN

int _orig_main(int argc, char **argv);

#endif  // WRAP_MAIN

#ifdef RDCYCLE

static inline uint64_t get_cycle() {
    uint64_t cycle;
    asm volatile ("rdcycle %0" : "=r" (cycle));
    return cycle;
}

#ifdef WRAP_MAIN

int __wrap_main(int argc, char **argv) {
    uint64_t start_time, end_time;

    start_time = get_cycle();

    int ret = __real_main(argc, argv);
    
    end_time = get_cycle();
    
    printf("Execution time: %llu cycles\n", end_time - start_time);

    return ret;
}

#else   // no WRAP_MAIN

int main(int argc, char **argv) {
    uint64_t start_time, end_time;

    start_time = get_cycle();

    int ret = _orig_main(argc, argv);
    
    end_time = get_cycle();
    
    printf("Execution time: %llu cycles\n", end_time - start_time);

    return 0;
}

#endif  // WRAP_MAIN

#else   // no RDCYCLE

#ifdef RV32ASM

struct timespec {
    __time_t tv_sec;
    long tv_nsec;
};

#else

#include <time.h>

#endif

static inline struct timespec get_timespec() {
    struct timespec ts;
#ifdef RV32ASM
    asm volatile (
        "li a7, 403\n"
        "li a0, 1\n"
        "mv a1, %0\n"
        "ecall\n"
        :
        : "r"(&ts)
        : "a0", "a1", "a7"
    );
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return ts;
}

#ifdef WRAP_MAIN

int __wrap_main(int argc, char **argv) {
    struct timespec start_ts, end_ts;

    start_ts = get_timespec();
    int ret = __real_main(argc, argv);
    end_ts = get_timespec();
    
    long diff_sec, diff_nsec;
    if ((end_ts.tv_nsec - start_ts.tv_nsec) < 0) {
        diff_sec = end_ts.tv_sec - start_ts.tv_sec - 1;
        diff_nsec = end_ts.tv_nsec - start_ts.tv_nsec + 1000000000;
    } else {
        diff_sec = end_ts.tv_sec - start_ts.tv_sec;
        diff_nsec = end_ts.tv_nsec - start_ts.tv_nsec;
    }
    printf("Execution time: %ld s %ld ns\n", diff_sec, diff_nsec);

    return ret;
}

#else   // no WRAP_MAIN

int main(int argc, char **argv) {
    struct timespec start_ts, end_ts;

    start_ts = get_timespec();
    int ret = _orig_main(argc, argv);
    end_ts = get_timespec();
    
    long diff_sec, diff_nsec;
    if ((end_ts.tv_nsec - start_ts.tv_nsec) < 0) {
        diff_sec = end_ts.tv_sec - start_ts.tv_sec - 1;
        diff_nsec = end_ts.tv_nsec - start_ts.tv_nsec + 1000000000;
    } else {
        diff_sec = end_ts.tv_sec - start_ts.tv_sec;
        diff_nsec = end_ts.tv_nsec - start_ts.tv_nsec;
    }
    printf("Execution time: %ld s %ld ns\n", diff_sec, diff_nsec);

    return 0;
}

#endif  // WRAP_MAIN

#endif  // RDCYCLE