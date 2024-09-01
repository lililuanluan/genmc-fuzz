#!/bin/bash

set -e

DIR=${PWD}
echo $DIR
cd ..
make -j8
cd $DIR

# TOOL=../genmcOrigin
TOOL=../genmc
# ESTIMATE=16620200       
#  --------- random exploration
# Estimation blocked: 16619722
# Estimation complete: 478
# uniq: 273667
# ---------- verification mode
# Number of complete executions explored: 7960
# Number of blocked executions seen: 288930   # total: 296890
# Total wall-clock time: 24.82s
# -print-estimation-stats

BENCH_DIR=/root/genmc-benchmarks
ESTIMATE=10000   # rand 4907
# ESTIMATE=1000

for i in {0..3}; do
  export MUTATION=${i}
  export Print=0

  for SEED in {2021..2025}; do
    echo "Running tests with MUTATION=${MUTATION} and SEED=${SEED}"

    # echo "----- testing dekker mutation = ${i}, seed = ${SEED} -----"
    # TEST=dekker-s${SEED} ${TOOL} --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -disable-sr -disable-ipr -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- ~/genmc-benchmarks/benchmarks/genmc/dekker_f/variants/dekker_f0.c
    echo "----- testing linuxrwlock mutation = ${i}, seed = ${SEED} -----"
    TEST=linuxrwlock-s${SEED} ${TOOL} --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -disable-sr -disable-ipr -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- -DCONFIG_RWLOCK_READERS=0 -DCONFIG_RWLOCK_WRITERS=0 -DCONFIG_RWLOCK_RDWR=2 ~/genmc-benchmarks/benchmarks/genmc/linuxrwlocks/variants/linuxrwlocks0.c

    echo "----- testing ttaslock mutation = ${i}, seed = ${SEED} -----"
    TEST=ttaslock-s${SEED} ${TOOL} --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -disable-sr -disable-ipr -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- -DN=6 ~/genmc-benchmarks/benchmarks/genmc/ttaslock-opt/variants/ttaslock0.c

    echo "----- testing big00 mutation = ${i}, seed = ${SEED} -----"
    TEST=big00-s${SEED} ${TOOL} --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- ${BENCH_DIR}/benchmarks/genmc/big0/variants/big00.c 

    echo "----- testing buf_ring mutation = ${i}, seed = ${SEED} -----"
    TEST=buf_ring-s${SEED} ${TOOL} --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- ${BENCH_DIR}/benchmarks/genmc/buf_ring/variants/buf_ring0.c

    echo "----- testing fib mutation = ${i}, seed = ${SEED} -----"
    TEST=fib-s${SEED} ${TOOL} --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- ${BENCH_DIR}/benchmarks/genmc/fib_bench/variants/fib_bench0.c 

    echo "----- testing mpmc-queue mutation = ${i}, seed = ${SEED} -----"
    TEST=mpmc-queue-s${SEED} ${TOOL} --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- -DCONFIG_MPMC_WRITERS=2 -DCONFIG_MPMC_READERS=4 -DCONFIG_MPMC_RDWR=0 ${BENCH_DIR}/benchmarks/genmc/mpmc-queue/variants/mpmc-queue.c

    echo "----- testing szymanski mutation = ${i}, seed = ${SEED} -----"
    TEST=szymanski-s${SEED} ${TOOL} --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- ${BENCH_DIR}/benchmarks/genmc/szymanski/variants/szymanski0.c  
  done
done


python3 ./stats.py