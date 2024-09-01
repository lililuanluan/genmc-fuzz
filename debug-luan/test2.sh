#!/bin/bash

set -e

DIR=${PWD}
echo $DIR
cd ..
make -j8
cd $DIR

export MUTATION=3
export Print=0

TOOL=../genmcOrigin

BENCH_DIR=/root/genmc-benchmarks
ESTIMATE=1000

TEST=ms_queue  ${TOOL} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -estimation-threshold=0 -schedule-policy=arbitrary -- -DCONFIG_QUEUE_READERS=3 -DCONFIG_QUEUE_WRITERS=3 ${BENCH_DIR}/benchmarks/genmc/ms-queue-dynamic/variants/ms_queue.c

# TEST=ms_queue  ${TOOL} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -estimation-threshold=0 -schedule-policy=arbitrary -- -DCONFIG_QUEUE_READERS=3 -DCONFIG_QUEUE_WRITERS=3 ${BENCH_DIR}/benchmarks/genmc/ms-queue/variants/main0.c

# TEST=long-race  ${TOOL} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary  ${DIR}/bench/long-race.c

# ${TOOL} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -estimation-threshold=0 -schedule-policy=arbitrary -- -DCONFIG_QUEUE_READERS=3 -DCONFIG_QUEUE_WRITERS=3 ${DIR}/test.c