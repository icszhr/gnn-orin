#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <functional>

#define BLOCK_SIZE 256  // CUDA block size
#define NUM_BLOCKS 4096 // Number of blocks for 1GB of data

const size_t CHUNK_SIZE = 1L * 1024 * 1024 * 1024; // 1GB
const size_t TOTAL_SIZE = 16L * 1024 * 1024 * 1024; // 16GB

__global__ void process_data(int *data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] + data[idx]; // Example operation: squaring the data
    }
}

// Helper function to measure execution time
void measure_performance(const char* description, const std::function<void()>& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << description << " took " << elapsed.count() << " seconds.\n";
}

void create_random_data_file(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(0, 100);
    
    for (size_t i = 0; i < TOTAL_SIZE / sizeof(int); ++i) {
        int random_value = distribution(generator);
        file.write(reinterpret_cast<const char*>(&random_value), sizeof(int));
    }
    file.close();
}

void test_with_cudaMallocManaged(int* mapped_data) {
    for (int i = 0; i < TOTAL_SIZE / CHUNK_SIZE; ++i) {
        // 启动CUDA内核，直接访问共享的统一内存
        int* device_data;
        cudaMallocManaged(&device_data, CHUNK_SIZE);
        cudaMemcpy(device_data, mapped_data + i * (CHUNK_SIZE / sizeof(int)), CHUNK_SIZE, cudaMemcpyHostToDevice);
        process_data<<<NUM_BLOCKS, BLOCK_SIZE>>>(device_data, CHUNK_SIZE / sizeof(int));
        cudaDeviceSynchronize();
        cudaFree(device_data);
    }
}


void test_with_cudaMalloc(int* mapped_data) {
    for (int i = 0; i < TOTAL_SIZE / CHUNK_SIZE; ++i) {
        int* device_data;
        cudaMalloc(&device_data, CHUNK_SIZE);
        cudaMemcpy(device_data, mapped_data + i * (CHUNK_SIZE / sizeof(int)), CHUNK_SIZE, cudaMemcpyHostToDevice);
        
        process_data<<<NUM_BLOCKS, BLOCK_SIZE>>>(device_data, CHUNK_SIZE / sizeof(int));
        cudaDeviceSynchronize();
        
        cudaFree(device_data);
    }
}

void test_with_pinmemory(int* mapped_data) {
    for (int i = 0; i < TOTAL_SIZE / CHUNK_SIZE; ++i) {
        cudaError_t err = cudaHostRegister(mapped_data + i * (CHUNK_SIZE / sizeof(int)), CHUNK_SIZE, cudaHostRegisterMapped);
        
        process_data<<<NUM_BLOCKS, BLOCK_SIZE>>>(mapped_data + i * (CHUNK_SIZE / sizeof(int)), CHUNK_SIZE / sizeof(int));
        cudaDeviceSynchronize();
    }
}

int main() {
    const char* filename = "16GB_random_data.bin";
    // create_random_data_file(filename);
    
    int fd = open(filename, O_RDWR | O_DIRECT);
    int* mapped_data = static_cast<int*>(mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);

    if (mapped_data == MAP_FAILED) {
        std::cerr << "Error: mmap failed." << std::endl;
        return 1;
    }

    // Run performance tests
    
    measure_performance("Using cudaMallocManaged", [&]() { test_with_cudaMallocManaged(mapped_data); });

    munmap(mapped_data, TOTAL_SIZE);
    return 0;
}
