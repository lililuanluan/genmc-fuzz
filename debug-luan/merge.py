import os
import glob
import re
# 定义数据文件所在目录
data_dir = "out"

# 定义测试用例名称和策略
tests = ["ms_queue", "ttaslock", "linuxrwlocks", "buf_ring", "treiber_stack", "mpmc-queue"]#, "szymanski"
strategies = ["rnd", "fz1", "fz2", "fz3"]

# 用于存储合并结果
merged_results = {test: {strategy: {} for strategy in strategies} for test in tests}
# merged_results = [f for f in merged_results if re.compile(r"-s(202[0-4])-\w+\.txt").search(f)]

# 读取每个测试用例文件
for test in tests:
    for strategy in strategies:
        # 找到所有相关文件
        files = glob.glob(os.path.join(data_dir, f"{test}-s*-{strategy}.txt"))
        
        # 用于存储所有执行次数的分数
        score_dict = {}

        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    execution_count, score = map(int, line.split())
                    if execution_count not in score_dict:
                        score_dict[execution_count] = []
                    score_dict[execution_count].append(score)

        # 计算平均得分
        for execution_count, scores in score_dict.items():
            avg_score = sum(scores) / len(scores)
            merged_results[test][strategy][execution_count] = avg_score

# 将合并结果写入新文件
for test, strategies in merged_results.items():
    for strategy, results in strategies.items():
        output_file = os.path.join(data_dir, f"{test}-{strategy}.txt")
        with open(output_file, 'w') as f:
            for execution_count, avg_score in sorted(results.items()):
                f.write(f"{execution_count} {avg_score:.2f}\n")
