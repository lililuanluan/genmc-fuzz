import os
import subprocess
import random

def run_command(command, shell=True):
    try:
        result = subprocess.run(command, shell=shell, check=True, text=True, capture_output=True)
        return result
    except subprocess.CalledProcessError as e:
        # print(e.stdout)
        # print(e.stderr)
        raise
    
TEST_DIRS = {
    "ttaslock": "benchmarks/genmc/ttaslock-opt/variants/ttaslock0.c",
    "ms_queue": "benchmarks/genmc/ms-queue-dynamic/variants/ms_queue.c",
    "big00": "benchmarks/genmc/big0/variants/big00.c",
    "buf_ring": "benchmarks/genmc/buf_ring/variants/buf_ring0.c",
    "fib": "benchmarks/genmc/fib_bench/variants/fib_bench0.c",
    "mpmc-queue": "benchmarks/genmc/mpmc-queue/variants/mpmc-queue.c",
    "szymanski": "benchmarks/genmc/szymanski/variants/szymanski0.c",
    
}
    
def run(TEST, TOOL, BENCH_DIR, TEST_DIR, ESTIMATE, MUTATION, Print, ARGS, SEED):
    os.environ["MUTATION"] = str(MUTATION)
    os.environ["Print"] = str(Print)
    os.environ["TEST"] = TEST
    run_command(f"{TOOL} --schedule-seed={SEED} -estimation-min={ESTIMATE} -estimation-max={ESTIMATE} {ARGS} {BENCH_DIR}/{TEST_DIR}")

def run_all():
    TOOL = "../genmc"
    SEED = 2024
    BENCH_DIR = "/root/genmc-benchmarks"
    ESTIMATE = 10000
    MUTATION = 3

    
    test = "ttaslock"
    print(f"----- testing {test} -----")
    run(TEST=test, TOOL=TOOL, BENCH_DIR=BENCH_DIR, TEST_DIR = TEST_DIRS[test], ESTIMATE=ESTIMATE, MUTATION=MUTATION, Print=0, ARGS="-disable-sr -disable-ipr  -print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- -DN=6", SEED=SEED)
    
    test = "ms_queue"
    print(f"----- testing {test} -----")
    run(TEST=test, TOOL=TOOL, BENCH_DIR=BENCH_DIR, TEST_DIR = TEST_DIRS[test], ESTIMATE=ESTIMATE, MUTATION=MUTATION, Print=0, ARGS="-print-estimation-stats -estimation-threshold=0 -schedule-policy=arbitrary -- -DCONFIG_QUEUE_READERS=3 -DCONFIG_QUEUE_WRITERS=3", SEED=SEED)
    
    test = "big00"
    print(f"----- testing {test} -----")
    run(TEST=test, TOOL=TOOL, BENCH_DIR=BENCH_DIR, TEST_DIR = TEST_DIRS[test], ESTIMATE=ESTIMATE, MUTATION=MUTATION, Print=0, ARGS="--print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 --", SEED=SEED)
    
    test = "buf_ring"
    print(f"----- testing {test} -----")
    run(TEST=test, TOOL=TOOL, BENCH_DIR=BENCH_DIR, TEST_DIR = TEST_DIRS[test], ESTIMATE=ESTIMATE, MUTATION=MUTATION, Print=0, ARGS="-print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 --", SEED=SEED)
    
    test = "fib"
    print(f"----- testing {test} -----")
    run(TEST=test, TOOL=TOOL, BENCH_DIR=BENCH_DIR, TEST_DIR = TEST_DIRS[test], ESTIMATE=ESTIMATE, MUTATION=MUTATION, Print=0, ARGS="-print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 --", SEED=SEED)
    
    test = "mpmc-queue"
    print(f"----- testing {test} -----")
    run(TEST=test, TOOL=TOOL, BENCH_DIR=BENCH_DIR, TEST_DIR = TEST_DIRS[test], ESTIMATE=ESTIMATE, MUTATION=MUTATION, Print=0, ARGS="-print-estimation-stats -schedule-policy=arbitrary -estimation-threshold=0 -- -DCONFIG_MPMC_WRITERS=2 -DCONFIG_MPMC_READERS=4 -DCONFIG_MPMC_RDWR=0", SEED=SEED)

def ms_queue():
    TOOL = "../genmc"
    SEED = 2024
    BENCH_DIR = "/root/genmc-benchmarks"
    ESTIMATE = 100000
    MUTATION = 3
    test = "ms_queue"
    stop = False
    cnt = 1
    for i in range(100):
        SEED = random.randint(0, 100000)
        try:
            run(TEST=test, TOOL=TOOL, BENCH_DIR=BENCH_DIR, TEST_DIR = TEST_DIRS[test], ESTIMATE=ESTIMATE, MUTATION=MUTATION, Print=0, ARGS="-print-estimation-stats -estimation-threshold=0 -schedule-policy=arbitrary -- -DCONFIG_QUEUE_READERS=3 -DCONFIG_QUEUE_WRITERS=3", SEED=SEED)
            print(f"success: {SEED}")
            cnt += 1
            if cnt > 5:
                return
        except:
            print(f"failed: {SEED}")
            pass

def main():
    DIR = os.getcwd()
    # print(DIR)
    
    os.chdir("..")
    run_command("make -j8")
    os.chdir(DIR)
    ms_queue()
    
    
    



if __name__ == "__main__":
    main()
