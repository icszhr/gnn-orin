// um.cu

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

    float *a, *b, *c;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        compute<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, NUM_ELEMENTS);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Unified Memory Average Execution Time: " << (duration.count() / 50) << " ms" << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
