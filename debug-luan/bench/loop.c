// genmc 处理无限的循环
#include <stdlib.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdbool.h>
#include <assert.h>
#include <stdatomic.h>

atomic_int x;
atomic_int y;

atomic_int cnt;

#define THRESHOLD 1

void* thread_1(void* arg) {
    (void)arg;
    // 下面这个循环，genmc会执行 spin-assume 
    // 如果一个完整循环没有可见的side effect

    // 这段代码，不断尝试将x从0设置为1
    // atomic_compare_exchange_strong 比较x的当前值是否与r相等，如果相等则将x设置为1，返回true
    // 如果不相等则将r设置为x的当前值，并返回false

    // 这个函数，在不相等的时候并不会改变x的值，所以没有副作用，r被重置了
    int r;
    while (!atomic_compare_exchange_strong(&x, &r, 1)) {
        r = 0;
        atomic_fetch_add(&cnt, 1);
        if (atomic_load_explicit(&cnt, memory_order_relaxed) > THRESHOLD) assert(0);
    }

    return NULL;
}


void* thread_2(void* arg) {
    (void)arg;
    int r = 1;
    while (!atomic_compare_exchange_strong(&x, &r, 0)) {
        r = 1;
        atomic_fetch_sub(&cnt, 1);
        if (atomic_load_explicit(&cnt, memory_order_relaxed) > THRESHOLD) return NULL;
    }
    return NULL;
}


int main() {
    pthread_t t1, t2;
    if (pthread_create(&t1, NULL, thread_1, NULL))
        abort();
    if (pthread_create(&t2, NULL, thread_2, NULL))
        abort();
    return 0;
}

