#!/bin/bash
# set -e
DIR=${PWD}
# echo $DIR
cd .. 
make -j8 > /dev/null
cd $DIR

TOOL=../genmc

BENCH_DIR=/root/genmc-benchmarks
ESTIMATE=10000000

# N=8

verify() {
    local N=$1
    # echo "--------mutation: verification---------"
    echo "verification"
../genmcOrigin -print-estimation-stats --disable-estimation --schedule-seed=${SEED} -- -DN=${N} fuzz.c 2>/dev/null | grep -E "Number of complete executions explored"
}

verify_t() {
    local N=$1
    # echo "--------mutation: verification---------"
    echo "verification"
    for SEED in {2010..2019}; do
        ../genmcOrigin -print-estimation-stats --disable-estimation --schedule-seed=${SEED} -- -DN=${N} fuzz.c 2>/dev/null | grep -E "Total wall-clock time:|Number of complete executions explored"
    done
}

get_mutation_name() {
    local i=$1
    case $i in
        0)
            echo "random"
            ;;
        1)
            echo "revisit"
            ;;
        25)
            echo "minimal"
            ;;
        85)
            echo "maximal"
            ;;
    esac
}


fuzz() {
    local N=$1
    for i in 0 1 25 85; do
        mutation_name=$(get_mutation_name $i)
        # echo "--------mutation: ${mutation_name}---------"
        echo "${mutation_name}"
        export MUTATION=${i}
        export Print=0
        for SEED in {2010..2020}; do
            TEST=fuzz ${TOOL} --schedule-seed=${SEED} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -disable-sr -disable-ipr -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0  -- -DN=${N} ./fuzz.c 2>/dev/null | grep "first"
        done
    done    
}

fuzz_t() {
    local N=$1
    # for i in  1 25 85; do
    for i in 0 1 25 85; do
        mutation_name=$(get_mutation_name $i)
        # echo "--------mutation: ${mutation_name}---------"
        echo "${mutation_name}"
        export MUTATION=${i}
        export Print=0
        for SEED in {2010..2019}; do
            TEST=fuzz ${TOOL} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -disable-sr -disable-ipr -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0  -- -DN=${N} ./fuzz.c 2>/dev/null | grep -E "Estimatioin time elapsed|first"
        done
    done    
}


test_x1() {    
    export HALT_ON_ERROR=1
    for N in {5..15}; do
        echo "N=${N}"
        echo "verification"
        ../genmcOrigin -print-estimation-stats --disable-estimation  -- -DN=${N} fuzz.c 2>/dev/null | grep -E "Total wall-clock time:|Number of complete executions explored" 


        for i in 0 1 25 85; do
            mutation_name=$(get_mutation_name $i)
            echo "${mutation_name}"
            export MUTATION=${i}
            export Print=0
            for x in {1..10}; do
            TEST=fuzz ${TOOL} -estimation-min=${ESTIMATE} -estimation-max=${ESTIMATE} -disable-sr -disable-ipr -print-estimation-stats -estimation-threshold=0  -- -DN=${N} ./fuzz.c 2>/dev/null | grep -E "Estimatioin time elapsed|first"
            done           
        done
    done    
}

run() {
    export HALT_ON_ERROR=1
    for N in {5..10}; do
        echo "N=${N}"
        verify ${N}
        fuzz ${N}
    done
}

run_t() {
    export HALT_ON_ERROR=1
    for N in 10; do
        echo "N=${N}"
        
        fuzz_t ${N}
        verify_t ${N}
    done
}

run_t
# test_x1 