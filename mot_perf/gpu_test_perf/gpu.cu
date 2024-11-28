// gpu.cu

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

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        compute<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, NUM_ELEMENTS);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "GPU Memory Average Execution Time: " << (duration.count() / 50) << " ms" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

