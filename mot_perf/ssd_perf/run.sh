#!/bin/bash

# 检查是否存在 16GB_random_data.bin 文件
if [ ! -f "16GB_random_data.bin" ]; then
    echo "Generating 16GB_random_data.bin..."
    nvcc create_file.cu -o create_file && ./create_file
    if [ $? -ne 0 ]; then
        echo "Failed to create 16GB_random_data.bin"
        exit 1
    fi
else
    echo "16GB_random_data.bin exists, skipping creation."
fi

# 编译其他文件
nvcc perfpin.cu -o perfpin && nvcc perfum.cu -o perfum && nvcc perfcpy.cu -o perfcpy
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Compilation successful. Running tests..."

# 执行测试
./perfpin | tee perfpin.txt
./perfum | tee perfum.txt
./perfcpy | tee perfcpy.txt
