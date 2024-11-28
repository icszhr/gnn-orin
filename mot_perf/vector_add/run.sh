#!/bin/bash

# 编译所有文件
nvcc perf.cu -o perf
nvcc perf2.cu -o perf2
nvcc perf3.cu -o perf3

# 检查是否所有文件都编译成功
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Compilation successful. Running tests..."

# 执行每个测试并将输出保存到相应文件
echo "Running perf..."
./perf | tee perf.txt

echo "Running perf2..."
./perf2 | tee perf2.txt

echo "Running perf3..."
./perf3 | tee perf3.txt


