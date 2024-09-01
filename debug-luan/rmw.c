#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <assert.h>

atomic_int shared_variable;
#define N 1
void* thread1_func(void* arg) {
    for (int i = 0; i < N; i++) {
        int temp = atomic_fetch_add(&shared_variable, 1);        
    }
    return NULL;
}
void* thread2_func(void* arg) {
    for (int i = 0; i < N; i++) {
        int temp = atomic_fetch_add(&shared_variable, 2);
    }
    return NULL;
}

int main() {
    atomic_init(&shared_variable, 0);

    pthread_t thread1, thread2;
    pthread_create(&thread1, NULL, thread1_func, NULL);
    pthread_create(&thread2, NULL, thread2_func, NULL);


    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    // assert(shared_variable == 3*N);

    return 0;
}
