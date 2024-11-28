// ori.cu

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define NUM_ELEMENTS 1000000000

__global__ void compute(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * 2.0f + b[idx];
    }
}

int main() {
    size_t size = NUM_ELEMENTS * sizeof(float);

    // 分配 CPU 内存
    float *a = new float[NUM_ELEMENTS];
    float *b = new float[NUM_ELEMENTS];
    float *c = new float[NUM_ELEMENTS];

    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配 GPU 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 50; ++i) {
        // 每次迭代都将数据从 CPU 复制到 GPU
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        // 执行 GPU 计算
        compute<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, NUM_ELEMENTS);
        cudaDeviceSynchronize();

        // 将计算结果从 GPU 复制回 CPU
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "CPU-GPU Transfer and Computation Average Execution Time: " << (duration.count() / 50) << " ms" << std::endl;

    // 释放内存
    delete[] a;
    delete[] b;
    delete[] c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
