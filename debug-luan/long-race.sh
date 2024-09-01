#!/bin/bash

set -e

DIR=${PWD}
echo $DIR
cd ..
make -j8
cd $DIR

ESTIMATE=100000

export MUTATION=2
export Print=0


# TEST=long-race  ../genmc -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 ./bench/long-race.c

TEST=loop  ../genmc -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 ./bench/loop.c