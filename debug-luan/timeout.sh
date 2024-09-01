#!/bin/bash

set -e

DIR=${PWD}
# echo $DIR
cd ..
make -j8   > /dev/null
# make
cd $DIR

TOOL=../genmc
BENCH_DIR=/root/genmc-benchmarks




run_test_case() {
  local test_name=$1
  local tool_options=$2
  local source_file=$3
  local extra_options=$4
  export SAMPLE_TIME=0   
  export TIME_OUT_SEC=60 

    
    
    # export REWARDF=2

  # for SEED in {2022..2025}; do
  for SEED in 2020; do
    for mut in 25 85 ; do  # 1不行 
    export MUTATION=${mut}
    # for SEED in 2021; do
      for bfsz in 200 ; do
        for rwd in 3; do
          # export PBUF_SIZE=${bfsz}
          echo ""
          # echo ">>> Running ${test_name} with MUTATION=${MUTATION} and SEED=${SEED} REWARD=${rwd} BUF_SIZE=${bfsz} PUSHNUM=${PUSHNUM}"
          echo ">>> Running ${test_name} with MUTATION=${MUTATION} and SEED=${SEED}"
          PBUF_SIZE=${bfsz} REWARDF=${rwd} TEST=${test_name}-t${TIME_OUT_SEC} ${TOOL} --disable-mm-detector  -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} ${tool_options} -print-estimation-stats --schedule-seed=${SEED} -schedule-policy=arbitrary -estimation-threshold=0 -- ${extra_options} ${source_file} 2>&1 # | grep -E "Estimatioin time elapsed"
        done
      done
    #    | grep "Estimatioin time elapsed"
    #   2>&1 | grep "Estimatioin time elapsed"
      
      
    done
  done
}


run() {
    ESTIMATE=10000000
    # export NO_SAVE_COVERAGE=1
    set +e
    export Print=0
    

    run_test_case "buf_ring" "" "${BENCH_DIR}/benchmarks/genmc/buf_ring/variants/buf_ring0.c" "-DNTHREADS=5"
    run_test_case "linuxrwlocks" "-disable-sr -disable-race-detection -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/linuxrwlocks/variants/linuxrwlocks0.c" "-DCONFIG_RWLOCK_READERS=2 -DCONFIG_RWLOCK_WRITERS=1 -DCONFIG_RWLOCK_RDWR=2"
    run_test_case "ms_queue" "" "/root/genmc-benchmarks/benchmarks/genmc/ms-queue-dynamic/variants/ms_queue.c" "-DCONFIG_QUEUE_READERS=3 -DCONFIG_QUEUE_WRITERS=4"
    run_test_case "mpmc-queue" "" "${BENCH_DIR}/benchmarks/genmc/mpmc-queue/variants/mpmc-queue.c" "-DCONFIG_MPMC_WRITERS=2 -DCONFIG_MPMC_READERS=4 -DCONFIG_MPMC_RDWR=0"
    run_test_case "ttaslock" "-disable-sr -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/ttaslock-opt/variants/ttaslock0.c" "-DN=6"
    run_test_case "treiber_stack" "-disable-sr -disable-race-detection -disable-ipr" "${BENCH_DIR}/benchmarks/genmc/treiber-stack-dynamic/variants/main0.c" "-DCONFIG_STACK_READERS=2 -DCONFIG_STACK_WRITERS=2 -DCONFIG_STACK_RDWR=2"
}



run