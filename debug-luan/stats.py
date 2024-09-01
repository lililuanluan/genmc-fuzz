import os

# 定义数据文件所在目录
data_dir = "out"

# 定义测试用例名称和策略
# tests = ["ms_queue", "ttaslock", "big00", "buf_ring", "fib", "mpmc-queue", "szymanski"]
tests = ["ms_queue", "ttaslock", "linuxrwlocks", "buf_ring", "treiber_stack", "mpmc-queue"]
strategies = ["rnd", "fz1", "fz2", "fz3"]

# 遍历每个测试用例和策略
for test in tests:
    print(" ------")
    for strategy in strategies:
        # 生成合并文件的路径
        merged_file = os.path.join(data_dir, f"{test}-{strategy}.txt")

        # 检查文件是否存在
        if os.path.exists(merged_file):
            with open(merged_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # 获取最后一行并提取分数和执行次数
                    last_line = lines[-1].strip()
                    execution_count, score = last_line.split()
                    # 打印格式为 "Test: test名, Mutation: mutation名, Average: 分数/次数"
                    print(f"Test: {test}, Mutation: {strategy}, Average: {score}/{execution_count}")
                else:
                    print(f"No data in {merged_file}.")
        else:
            print(f"{merged_file} does not exist.")
