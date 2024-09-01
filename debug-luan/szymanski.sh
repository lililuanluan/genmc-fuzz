#!/bin/bash

DIR=${PWD}
echo $DIR
cd ..
make -j8
cd $DIR

TOOL=../genmc
BENCH_DIR=/root/genmc-benchmarks
ESTIMATE=1000

export MUTATION=3
export Print=1
SEED=2024

TEST=szymanski   ${TOOL} -estimation-min=${ESTIMATE} --schedule-seed=${SEED} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- ${BENCH_DIR}/benchmarks/genmc/szymanski/variants/szymanski0.c  