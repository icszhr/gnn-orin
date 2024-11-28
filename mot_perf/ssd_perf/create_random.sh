#!/bin/bash

# 文件名和大小
filename="16GB_random_data.bin"
size=16G

# 使用 /dev/urandom 生成随机数据并写入文件
dd if=/dev/urandom of=$filename bs=1M count=16384 status=progress

echo "文件 $filename 已创建，大小 $size"

