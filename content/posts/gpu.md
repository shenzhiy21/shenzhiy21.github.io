+++
title = 'CUDA Notes'
date = 2025-09-16T00:41:54+08:00
draft = false
math = true
tags = ['note']
categories = ['note', 'system', 'parallel', 'memory']
summary = "Just tracing the process of learning CUDA programming"
+++

## Resources

- [GPU-Mode Lectures](https://github.com/gpu-mode/lectures)
- Programming Massively Parallel Processors

## Programming Massively Parallel Processors (PMPP) TakeAways

### Chapter 1. Introduction

Design philosophies for CPU/GPU design:

- CPU: latency oriented, optimized for sequential code performance
- GPU: throughput oriented (floating-point calculations and memory access throughput, "parallel")

Two types of bounds

- Memory bound: applications where execution speed is limited by memory access latency and/or throughput
- Compute bound: ... limited by the number of instructions performed per byte of data

### Chapter 2. Heterogeneous data parallel computing

Structure of CUDA C program: a host (CPU) and one or more deviced (GPU).
Device code is marked with special CUDA C keywords.
Device code includes *kernels* whose code is executed in a data-parallel manner.

When a kernel function is called, a large number of **threads** are launched on a device.
All the threads that are launched by a kernel can be collectively called a **grid**.
Each grid is organized as an array of thread blocks, simplified as **blocks**.
All blocks are of the same size (up to 1024 threads).

Allocating memories in CUDA C:
- `cudaMalloc()`: allocates object in device global memory
  - Address of a pointer to the allocated object
  - Size of allocated object in terms of byte
- `cudaFree()`: frees object from device global memory
  - Pointer to the freed object
- `cudaMemcpy()`: memory data transfer
  - Pointer to destination
  - Pointer to source
  - Number of bytes copied
  - Type of transfer (`cudaMemcpyHostToDevice`, ...)

An simplified example:

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // kernel invocation code
    ...

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

Built-in variables in CUDA kernel:
- `blockDim`: the number of threads in a block. It is a struct with three unsigned integers (x, y, z) to organize the
threads into a one-, two-, or three-dimensional array.
- `threadIdx`: distinguish each thread in a block. Also (x, y, z).
- `blockIdx`: distinguish each block.

A unique global index for a thread in a one-dimensional layout is:

```c
int i = threadIdx.x + blockDim.x * blockIdx.x;
```

The kernel function for `vecAdd`:

```c
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

`__global__` is a CUDA-C-specific keyword to indicate that the following function is a kernel.

CUDA C keywords for function declaration:

| Keyword               | Call From        | Executed On | Executed By               |
|-----------------------|-----------------|-------------|---------------------------|
| `__host__` (default)  | Host            | Host        | Caller host thread        |
| `__global__`          | Host (or Device)| Device      | New grid of device threads|
| `__device__`          | Device          | Device      | Caller device thread      |

To call the kernel function:

```c
vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
```

The kernel function has two execution configuration parameters between `<<<` and `>>>`.
- The first: number of blocks in the grid
- The second: number of threads in each block

Compilation of CUDA C:
- NVCC (Nvidia C compiler)
- Host code: compiled with host's standard C/C++ compilers, run as traditional CPU process
- Device code: marked with CUDA keywords, compiled by NVCC (device just-in-time compiler)

