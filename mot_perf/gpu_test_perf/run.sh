#!/bin/bash

# 编译所有文件
nvcc pin.cu -o pin_test
nvcc um.cu -o um_test
nvcc gpu.cu -o gpu_test
nvcc ori.cu -o ori_test

# 检查是否所有文件都编译成功
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Compilation successful. Running tests..."

# 执行每个测试并将输出保存到相应文件
echo "Running Pinned Memory Test..."
./pin_test | tee pin_output.txt

echo "Running Unified Memory Test..."
./um_test | tee um_output.txt

echo "Running GPU Memory Test..."
./gpu_test | tee gpu_output.txt

echo "Running CPU-GPU Transfer Test..."
./ori_test | tee ori_output.txt

echo "All tests completed. Results are saved in pin_output.txt, um_output.txt, gpu_output.txt, and ori_output.txt."

