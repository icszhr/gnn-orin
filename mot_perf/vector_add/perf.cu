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

void cpuFill(float* A, float* B, int Num) {
    for (int i = 0; i < Num; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }
}

void evaluatePinnedMemory() {
    float *A, *B, *C;
    float *dA, *dB, *dC;

    auto start = std::chrono::high_resolution_clock::now();
    // Allocate pinned memory (host memory)
    cudaHostAlloc(&A, N * sizeof(float), cudaHostAllocDefault); // Pinned memory
    cudaHostAlloc(&B, N * sizeof(float), cudaHostAllocDefault); // Pinned memory
    cudaHostAlloc(&C, N * sizeof(float), cudaHostAllocDefault); // Pinned memory

    cpuFill(A, B, N); // Fill data in CPU

    // Get device pointers corresponding to the host memory
    cudaHostGetDevicePointer(&dA, A, 0); // Device pointer for A
    cudaHostGetDevicePointer(&dB, B, 0); // Device pointer for B
    cudaHostGetDevicePointer(&dC, C, 0); // Device pointer for C

    // Perform the kernel operation multiple times to assess the performance
    for (int i = 0; i < 10; i++) {
        vectorAdd<<<(N + 255) / 256, 256>>>(dA, dB, dC, N);
        cudaDeviceSynchronize(); // Ensure the computation is done
    }

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

    // Allocate unified memory
    auto start = std::chrono::high_resolution_clock::now();

    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));

    cpuFill(A, B, N); // Fill data in CPU

    // Perform the kernel operation multiple times to assess the performance
    for (int i = 0; i < 10; i++) {
        vectorAdd<<<(N + 255) / 256, 256>>>(A, B, C, N);
        cudaDeviceSynchronize(); // Ensure the computation is done
    }

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

    // Allocate memory for CPU and GPU
    auto start = std::chrono::high_resolution_clock::now();
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));
    
    cudaMalloc(&dA, N * sizeof(float)); // Device memory
    cudaMalloc(&dB, N * sizeof(float)); // Device memory
    cudaMalloc(&dC, N * sizeof(float)); // Device memory

    cpuFill(A, B, N); // Fill data in CPU

    // Perform the kernel operation multiple times to assess the performance
    for (int i = 0; i < 10; i++) {
        // Copy data from Host to Device
        cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, N * sizeof(float), cudaMemcpyHostToDevice);

        // Perform addition
        vectorAdd<<<(N + 255) / 256, 256>>>(dA, dB, dC, N);
        cudaDeviceSynchronize(); // Ensure the computation is done

        // Copy the result back to CPU
        cudaMemcpy(C, dC, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

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

    // Allocate memory for CPU and GPU
    auto start = std::chrono::high_resolution_clock::now();
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));
    
    cudaMalloc(&dA, N * sizeof(float)); // Device memory
    cudaMalloc(&dB, N * sizeof(float)); // Device memory
    cudaMalloc(&dC, N * sizeof(float)); // Device memory

    cpuFill(A, B, N); // Fill data in CPU

    // Perform the kernel operation multiple times to assess the performance
    for (int i = 0; i < 10; i++) {
        vectorAdd<<<(N + 255) / 256, 256>>>(dA, dB, dC, N);
        cudaDeviceSynchronize(); // Ensure the computation is done
    }

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
        // Evaluate Pure Computation (without data transfer)
    evaluatePureComputation();
        // Evaluate Unified Memory
    evaluateManagedMemory();
    
    // Evaluate Pinned Memory
    evaluatePinnedMemory();
    

    
    // Evaluate Memcpy Transfer
    evaluateMemcpy();
    
    // Evaluate Pure Computation (without data transfer)
    evaluatePureComputation();

    return 0;
}
