import torch
import time
import numpy as np
import mmap
import os

# Parameters
total_ram_gb = 32  # Total RAM on Jetson Orin in GB
mmap_extension_gb = 32  # Additional space via mmap for 64GB addressable space
num_elements = 1000000  # Number of elements for computations (adjust for testing)
page_size = mmap.PAGESIZE
file_size = mmap_extension_gb * 1024**3  # Size for mmap file to extend to 32GB

# Create a file to back the mmap for the additional 32GB virtual memory
with open("mmap_testfile", "wb") as f:
    f.write(b'\0' * file_size)

# Helper function to time a test
def time_test(operation):
    start = time.time()
    result = operation()
    return time.time() - start, result

# A function to simulate workload (a simple operation across memory)
def simulate_workload(tensor):
    tensor *= torch.sin(tensor)  # Example computation
    return tensor.sum().item()

# Regular mode: Explicitly transfer data between GPU and CPU, with potential paging
def regular_mode():
    cpu_tensor = torch.randn(num_elements, dtype=torch.float32)  # CPU memory
    gpu_tensor = torch.empty_like(cpu_tensor, device='cuda')  # Empty GPU tensor
    gpu_tensor.copy_(cpu_tensor)  # Explicit copy to GPU
    return simulate_workload(gpu_tensor)

# Unified memory mode: Uses CUDA Unified Memory to manage all RAM
def unified_memory_mode():
    tensor = torch.randn(num_elements, dtype=torch.float32, device='cuda')  # CUDA unified memory
    return simulate_workload(tensor)

# Pinned memory mode: Locks half of memory for direct GPU access, no swapping
def pinned_memory_mode():
    tensor = torch.randn(num_elements, dtype=torch.float32, pin_memory=True)  # Pinned memory for direct GPU access
    gpu_tensor = tensor.to('cuda', non_blocking=True)  # Move to GPU with non-blocking transfer
    return simulate_workload(gpu_tensor)

# Hybrid mode: Half of RAM is pinned, remaining managed by unified memory
def hybrid_mode():
    half_elements = num_elements // 2
    pinned_tensor = torch.randn(half_elements, dtype=torch.float32, pin_memory=True)  # Pinned memory
    unified_tensor = torch.randn(half_elements, dtype=torch.float32, device='cuda')  # Unified memory
    pinned_gpu = pinned_tensor.to('cuda', non_blocking=True)  # Move pinned memory to GPU
    return simulate_workload(pinned_gpu) + simulate_workload(unified_tensor)

# Running tests
print("Running regular mode test...")
time_regular, result_regular = time_test(regular_mode)

print("Running unified memory mode test...")
time_unified, result_unified = time_test(unified_memory_mode)

print("Running pinned memory mode test...")
time_pinned, result_pinned = time_test(pinned_memory_mode)

print("Running hybrid mode test...")
time_hybrid, result_hybrid = time_test(hybrid_mode)

# Print results
print(f"Regular Mode Time: {time_regular:.4f} seconds, Result: {result_regular}")
print(f"Unified Memory Mode Time: {time_unified:.4f} seconds, Result: {result_unified}")
print(f"Pinned Memory Mode Time: {time_pinned:.4f} seconds, Result: {result_pinned}")
print(f"Hybrid Mode Time: {time_hybrid:.4f} seconds, Result: {result_hybrid}")

# Cleanup mmap file
os.remove("mmap_testfile")
