#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring> // 添加此行

#define GB_5 (5 * 1024 * 1024 * 1024ULL) // 5 GB
#define ELEMENT_COUNT (GB_5 / sizeof(float)) // 5 GB 对应的 float 元素数量

void testCudaManagedToMmap() {
    float *managedMemory;
    int fd = open("managed_swap.bin", O_RDWR | O_CREAT, 0666);
    ftruncate(fd, GB_5);
    float *mmapMemory = (float *)mmap(nullptr, GB_5, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    cudaMallocManaged(&managedMemory, GB_5);

    // 初始化统一内存数据
    for (size_t i = 0; i < ELEMENT_COUNT; i++) {
        managedMemory[i] = static_cast<float>(rand() % 100);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // 将数据从统一内存复制到 mmap 磁盘交换空间
    memcpy(mmapMemory, managedMemory, GB_5);
    cudaFree(managedMemory);

    // 分配新的统一内存并从 mmap 空间读取数据
    cudaMallocManaged(&managedMemory, GB_5);
    memcpy(managedMemory, mmapMemory, GB_5);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "cudaMallocManaged to mmap swap time: " << duration.count() << " seconds." << std::endl;

    // 清理资源
    cudaFree(managedMemory);
    munmap(mmapMemory, GB_5);
    remove("managed_swap.bin");
}

void testCudaMallocToMmap() {
    float *deviceMemory, *hostMemory;
    int fd = open("malloc_swap.bin", O_RDWR | O_CREAT, 0666);
    ftruncate(fd, GB_5);
    float *mmapMemory = (float *)mmap(nullptr, GB_5, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    cudaMalloc(&deviceMemory, GB_5);
    hostMemory = (float *)malloc(GB_5);

    // 初始化 GPU 内存数据
    for (size_t i = 0; i < ELEMENT_COUNT; i++) {
        hostMemory[i] = static_cast<float>(rand() % 100);
    }
    cudaMemcpy(deviceMemory, hostMemory, GB_5, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    // 将数据从 GPU 内存复制到主机内存
    cudaMemcpy(hostMemory, deviceMemory, GB_5, cudaMemcpyDeviceToHost);

    // 将数据从主机内存复制到 mmap 磁盘交换空间
    memcpy(mmapMemory, hostMemory, GB_5);

    // 从 mmap 空间将数据复制回主机内存
    memcpy(hostMemory, mmapMemory, GB_5);

    // 将数据复制回另一块 GPU 内存
    cudaMemcpy(deviceMemory, hostMemory, GB_5, cudaMemcpyHostToDevice);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "cudaMalloc to mmap swap time: " << duration.count() << " seconds." << std::endl;

    // 清理资源
    free(hostMemory);
    cudaFree(deviceMemory);
    munmap(mmapMemory, GB_5);
    remove("malloc_swap.bin");
}

int main() {
    testCudaManagedToMmap();
    testCudaMallocToMmap();
    return 0;
}
