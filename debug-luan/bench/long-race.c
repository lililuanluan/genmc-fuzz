#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <assert.h>

static atomic_size_t sum = ATOMIC_VAR_INIT(0);
static atomic_size_t dif = ATOMIC_VAR_INIT(0);

#define ORDER memory_order_relaxed
// #define ORDER memory_order_seq_cst 

static void* sub_worker(void* arg) {
    if (atomic_load_explicit(&sum, ORDER) == 1) {
        atomic_fetch_sub_explicit(&dif, 1, ORDER);
        if (atomic_load_explicit(&sum, ORDER) == 2) {
            atomic_fetch_sub_explicit(&dif, 1, ORDER);
            if (atomic_load_explicit(&sum, ORDER) == 3) {
                atomic_fetch_sub_explicit(&dif, 1, ORDER);
                if (atomic_load_explicit(&sum, ORDER) == 4) {
                    atomic_fetch_sub_explicit(&dif, 1, ORDER);
                }
            }
        }
    }
    return NULL;
}

static void* add_worker(void* arg) {
    atomic_fetch_add_explicit(&sum, 1, ORDER);

    if (atomic_load_explicit(&dif, ORDER) == -1) {
        atomic_fetch_add_explicit(&sum, 1, ORDER);
        if (atomic_load_explicit(&dif, ORDER) == -2) {
            atomic_fetch_add_explicit(&sum, 1, ORDER);
            if (atomic_load_explicit(&dif, ORDER) == -3) {
                atomic_fetch_add_explicit(&sum, 1, ORDER);
                if (atomic_load_explicit(&dif, ORDER) == -4) {
                    fprintf(stderr, "Bug found\n");
                    assert(0);
                }
            }
        }
    }

    return NULL;
}

// requires: add -> sub -> add -> sub -> add -> sub ...

int main(int argc, char** argv) {
    pthread_t adder;
    pthread_t suber;

    pthread_create(&adder, NULL, add_worker, NULL);
    pthread_create(&suber, NULL, sub_worker, NULL);

    pthread_join(adder, NULL);
    pthread_join(suber, NULL);

    int s = atomic_load_explicit(&sum, ORDER);
    if (s > 3)
        printf("sum = %zu\n", s); // Output depends on race conditions
    return 0;
}
