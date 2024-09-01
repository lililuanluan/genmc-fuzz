#!/bin/bash

set -e

DIR=${PWD}
echo $DIR
cd ..
make -j8
cd $DIR

# TOOL=../genmcOrigin
TOOL=../genmc
SEED=2024

BENCH_DIR=/root/genmc-benchmarks
# ESTIMATE=100000   # rand 4907
ESTIMATE=10000


export MUTATION=4
export Print=1

# TEST=rmw   ${TOOL} --schedule-seed=${SEED} -disable-mm-detector -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- ./rmw.c