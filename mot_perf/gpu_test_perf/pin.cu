// pin.cu

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define NUM_ELEMENTS 1000000000  // 数据块大小

__global__ void compute(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * 2.0f + b[idx];
    }
}

int main() {
    size_t size = NUM_ELEMENTS * sizeof(float);

    // 使用 Pinned Memory 分配并获取 GPU 可访问的指针
    float *a, *b, *c;
    cudaHostAlloc((void**)&a, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&b, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&c, size, cudaHostAllocDefault);

    // 初始化数据
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 获取锁定内存的 GPU 地址
    float *d_a, *d_b, *d_c;
    cudaHostGetDevicePointer((void**)&d_a, a, 0);
    cudaHostGetDevicePointer((void**)&d_b, b, 0);
    cudaHostGetDevicePointer((void**)&d_c, c, 0);

    // 设置 CUDA 计算参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;

    // 记录计算时间
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        compute<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, NUM_ELEMENTS);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Pinned Memory Average Execution Time: " << (duration.count() / 50) << " ms" << std::endl;

    // 释放 Pinned Memory
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);

    return 0;
}
