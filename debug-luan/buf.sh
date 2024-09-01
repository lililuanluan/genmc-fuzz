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

    
    
    # export REWARDF=2

  # for SEED in {2022..2025}; do
  for SEED in 2024; do
    for mut in 1; do  # 1不行 
    export MUTATION=${mut}
    # for SEED in 2021; do
      for bfsz in 200 ; do
        for rwd in 3; do
          # export PBUF_SIZE=${bfsz}
          echo ""
          # echo ">>> Running ${test_name} with MUTATION=${MUTATION} and SEED=${SEED} REWARD=${rwd} BUF_SIZE=${bfsz} PUSHNUM=${PUSHNUM}"
          echo ">>> Running ${test_name} with MUTATION=${MUTATION} and SEED=${SEED} N=${N}"
          PBUF_SIZE=${bfsz} REWARDF=${rwd} TEST=${test_name}-s${SEED}-N${N}-p${PRIORITIZE_RF} ${TOOL} --disable-mm-detector  -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} ${tool_options} -print-estimation-stats --schedule-seed=${SEED} -schedule-policy=arbitrary -estimation-threshold=0 -- ${extra_options} ${source_file} 2>&1 # | grep -E "Estimatioin time elapsed"
        done
      done
    #    | grep "Estimatioin time elapsed"
    #   2>&1 | grep "Estimatioin time elapsed"
      
      
    done
  done
}


run_buf_ring() {
    ESTIMATE=100000
    
    # export NO_SAVE_COVERAGE=1
    set +e
    export Print=0
    
    SEED=2024
    test_name="buf_ring"

    for pri in 0; do 
      for n in 3 4 5 6; do 
        for mut in 0; do
        export MUTATION=${mut}
        N=${n}
        export PRIORITIZE_RF=${pri}
        echo ">>> Running ${test_name} with MUTATION=${MUTATION} and SEED=${SEED} N=${N}"
        TEST=${test_name}-s${SEED}-N${N}-p${PRIORITIZE_RF} ${TOOL} --disable-mm-detector  -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} ${tool_options} -print-estimation-stats --schedule-seed=${SEED} -schedule-policy=arbitrary -estimation-threshold=0 -- -DNTHREADS=${N} ${BENCH_DIR}/benchmarks/genmc/buf_ring/variants/buf_ring0.c 2>&1 
        done
      done
    done
}

run_mpmc_queue() {
    ESTIMATE=100000
    
    # export NO_SAVE_COVERAGE=1
    set +e
    export Print=0
    
    SEED=2024
    test_name="mpmc_queue"

    for pri in 0 1; do 
      for n in 3 4 5 6; do 
        for mut in 0 1 2 3; do
        export MUTATION=${mut}
        N=${n}
        export PRIORITIZE_RF=${pri}
        echo ">>> Running ${test_name} with MUTATION=${MUTATION} and SEED=${SEED} N=${N}"
        TEST=${test_name}-s${SEED}-N${N}-p${PRIORITIZE_RF} ${TOOL} --disable-mm-detector  -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} ${tool_options} -print-estimation-stats --schedule-seed=${SEED} -schedule-policy=arbitrary -estimation-threshold=0 -- -DCONFIG_MPMC_WRITERS=2 -DCONFIG_MPMC_READERS=${N} -DCONFIG_MPMC_RDWR=0 ${BENCH_DIR}/benchmarks/genmc/mpmc-queue/variants/mpmc-queue.c 2>&1 
        done
      done
    done
}

# run_buf_ring
run_mpmc_queue
# verify_buf_ring