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


int main() {
    const char* filename = "16GB_random_data.bin";
    create_random_data_file(filename);

    return 0;
}
