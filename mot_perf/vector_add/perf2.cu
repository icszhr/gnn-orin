#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 1024*1024 // Number of elements

// CUDA Kernel for element-wise addition
__global__ void vectorAdd(float* A, float* B, float* C, int Num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < Num) {
        C[idx] = A[idx] + B[idx];
    }
}

void evaluatePinnedMemory() {
    float *A, *B, *C;
    float *dA, *dB, *dC;

    // Start timing the entire operation (memory allocation + computation)
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate pinned memory (host memory)
    cudaHostAlloc(&A, N * sizeof(float), cudaHostAllocDefault); // Pinned memory
    cudaHostAlloc(&B, N * sizeof(float), cudaHostAllocDefault); // Pinned memory
    cudaHostAlloc(&C, N * sizeof(float), cudaHostAllocDefault); // Pinned memory

    // Get device pointers corresponding to the host memory
    cudaHostGetDevicePointer(&dA, A, 0); // Device pointer for A
    cudaHostGetDevicePointer(&dB, B, 0); // Device pointer for B
    cudaHostGetDevicePointer(&dC, C, 0); // Device pointer for C

    // Perform the kernel operation
    vectorAdd<<<(N + 255) / 256, 256>>>(dA, dB, dC, N);
    cudaDeviceSynchronize(); // Ensure the computation is done

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Pinned memory time: " << duration.count() << " seconds." << std::endl;

    // Free resources
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
}

void evaluateManagedMemory() {
    float *A, *B, *C;

    // Start timing the entire operation (memory allocation + computation)
    auto start = std::chrono::high_resolution_clock::now();

    // Allocate unified memory
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));

    // Perform the kernel operation
    vectorAdd<<<(N + 255) / 256, 256>>>(A, B, C, N);
    cudaDeviceSynchronize(); // Ensure the computation is done

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Unified memory time: " << duration.count() << " seconds." << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

void evaluateMemcpy() {
    float *A, *B, *C;
    float *dA, *dB, *dC;

    // Start timing the entire operation (memory allocation + computation)
    auto start = std::chrono::high_resolution_clock::now();

    // Allocate memory for CPU and GPU
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));

    cudaMalloc(&dA, N * sizeof(float)); // Device memory
    cudaMalloc(&dB, N * sizeof(float)); // Device memory
    cudaMalloc(&dC, N * sizeof(float)); // Device memory

    // Copy data from Host to Device
    cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform the kernel operation
    vectorAdd<<<(N + 255) / 256, 256>>>(dA, dB, dC, N);
    cudaDeviceSynchronize(); // Ensure the computation is done

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Memcpy transfer time: " << duration.count() << " seconds." << std::endl;

    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void evaluatePureComputation() {
    float *A, *B, *C;
    float *dA, *dB, *dC;

    // Start timing the entire operation (memory allocation + computation)
    auto start = std::chrono::high_resolution_clock::now();

    // Allocate memory for CPU and GPU
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));

    cudaMalloc(&dA, N * sizeof(float)); // Device memory
    cudaMalloc(&dB, N * sizeof(float)); // Device memory
    cudaMalloc(&dC, N * sizeof(float)); // Device memory

    // Perform the kernel operation
    vectorAdd<<<(N + 255) / 256, 256>>>(dA, dB, dC, N);
    cudaDeviceSynchronize(); // Ensure the computation is done

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Pure computation time: " << duration.count() << " seconds." << std::endl;

    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main() {
        // Evaluate Pure Computation (memory allocation + computation)
    evaluatePureComputation();
    // Evaluate Pure Computation (memory allocation + computation)
    evaluatePureComputation();
    

    
    // Evaluate Pinned Memory (memory allocation + computation)
    evaluatePinnedMemory();

        // Evaluate Unified Memory (memory allocation + computation)
    evaluateManagedMemory();
    
    // Evaluate Memcpy Transfer (memory allocation, copy + computation)
    evaluateMemcpy();

    return 0;
}
