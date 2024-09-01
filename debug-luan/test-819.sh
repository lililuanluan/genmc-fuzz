#!/bin/bash

set -e

DIR=${PWD}
# echo $DIR
cd ..
make -j8  > /dev/null
# make
cd $DIR


TOOL=../genmc

BENCH_DIR=/root/genmc-benchmarks


run_test_case() {
    local test_name=$1
    local tool_options=$2
    local source_file=$3
    local extra_options=$4

    # for mut in 85; do
    for mut in  25 85; do # various cuts with powersched
        export MUTATION=${mut}
        bfsz=200
        PUSHNUM=1
        for SEED in 2024; do
            echo ""
            echo ">>> Running ${test_name} with MUTATION=${MUTATION} and SEED=${SEED} REWARD=${rwd} BUF_SIZE=${bfsz} PUSHNUM=${PUSHNUM}"
            PBUF_SIZE=${bfsz} REWARDF=${rwd} TEST=${test_name}-s${SEED}-bf${PBUF_SIZE}-push${PUSHNUM} ${TOOL} --disable-mm-detector  --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} ${tool_options} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- ${extra_options} ${source_file}
        done
    done

}

# run_test_case test_name tool_options source_file extra_options

run_dekker() {
    run_test_case "dekker" "-disable-sr -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/dekker_f/variants/dekker_f0.c" "-DN=5"
}

# run_linuxrwlock() {
#   run_test_case "linuxrwlock" "-disable-sr -disable-ipr" " ${BENCH_DIR}/benchmarks/genmc/linuxrwlocks/variants/linuxrwlocks0.c" "-DCONFIG_RWLOCK_READERS=0 -DCONFIG_RWLOCK_WRITERS=0 -DCONFIG_RWLOCK_RDWR=2"
# }

run_ttaslock() {
  run_test_case "ttaslock" "-disable-sr -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/ttaslock-opt/variants/ttaslock0.c" "-DN=6"
}

run_ttaslock2() {
  run_test_case "ttaslock" "-disable-sr -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/ttaslock-opt/variants/ttaslock0.c" "-DN=3"
}

run_big00() {
  run_test_case "big00" "" "${BENCH_DIR}/benchmarks/genmc/big0/variants/big00.c"
}

run_buf_ring() {
  run_test_case "buf_ring" "" "${BENCH_DIR}/benchmarks/genmc/buf_ring/variants/buf_ring0.c"
}

run_fib() {
  run_test_case "fib" "" "${BENCH_DIR}/benchmarks/genmc/fib_bench/variants/fib_bench0.c"
}

run_mpmc_queue() {
  run_test_case "mpmc-queue" "" "${BENCH_DIR}/benchmarks/genmc/mpmc-queue/variants/mpmc-queue.c" "-DCONFIG_MPMC_WRITERS=2 -DCONFIG_MPMC_READERS=4 -DCONFIG_MPMC_RDWR=0"
}

run_szymanski() {
  run_test_case "szymanski" "-disable-sr -disable-race-detection -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/szymanski/variants/szymanski0.c" "-DN=10"
}

run_peterson() {
    run_test_case "peterson" "" "${BENCH_DIR}/benchmarks/genmc/peterson-sc/variants/peterson0.c"
}

run_msque() {
    run_test_case "ms_queue" "" "/root/genmc-benchmarks/benchmarks/genmc/ms-queue-dynamic/variants/ms_queue.c" "-DCONFIG_QUEUE_READERS=3 -DCONFIG_QUEUE_WRITERS=3"
}



run_linuxrwlocks() {
    echo "${BENCH_DIR}/benchmarks/genmc/linuxrwlocks/variants/linuxrwlocks0.c"
    run_test_case "linuxrwlocks" "-disable-sr -disable-race-detection -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/linuxrwlocks/variants/linuxrwlocks0.c" "-DCONFIG_RWLOCK_READERS=2 -DCONFIG_RWLOCK_WRITERS=1 -DCONFIG_RWLOCK_RDWR=2"
}

run_linuxrwlocks2() {
    echo "${BENCH_DIR}/benchmarks/genmc/linuxrwlocks/variants/linuxrwlocks0.c"
    run_test_case "linuxrwlocks" "-disable-sr -disable-race-detection -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/linuxrwlocks/variants/linuxrwlocks0.c" "-DCONFIG_RWLOCK_READERS=1 -DCONFIG_RWLOCK_WRITERS=1 -DCONFIG_RWLOCK_RDWR=1"
}

run_treiber() {
    run_test_case "treiber_stack" "-disable-sr -disable-race-detection -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/treiber-stack-dynamic/variants/main0.c" "-DCONFIG_STACK_READERS=2 -DCONFIG_STACK_WRITERS=2 -DCONFIG_STACK_RDWR=2"
}

run_treiber2() {
    run_test_case "treiber_stack" "-disable-sr -disable-race-detection -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/treiber-stack-dynamic/variants/main0.c" "-DCONFIG_STACK_READERS=1 -DCONFIG_STACK_WRITERS=1 -DCONFIG_STACK_RDWR=1"
}



test_rmw() {
    export Print=0
    # ESTIMATE=100
    ESTIMATE=100000 
    run_linuxrwlocks   
    run_treiber    
    run_buf_ring #   
    run_mpmc_queue   
    run_msque  

}

test_all() {
    export Print=0
    export PUSHNUM=5
    set +e
    # ESTIMATE=100
    ESTIMATE=100000 
    # run_linuxrwlocks   
    # run_treiber    
    run_buf_ring #   
    # run_mpmc_queue   
    # run_msque  
    # run_ttaslock
}

# 启用或禁用需要运行的测试
export Print=0
# ESTIMATE=100
ESTIMATE=100000 

# test_rmw
# test_all


# run_linuxrwlocks   
# run_linuxrwlocks2
run_treiber    
# run_treiber2
# run_ttaslock
# run_ttaslock2
# run_dekker
# run_buf_ring #   
# run_mpmc_queue   
# run_msque  

# python3 ./merge.py
# python3 ./stats.py
