+++
title = 'Write a Memory Allocator for PyTorch'
date = 2024-05-16T15:21:52+08:00
draft = false
math = true
tags = ['code', 'memory', 'system', 'algorithm', 'machine learning']
categories = ['code']
summary = "Write a memory allocator from scratch."

+++

## Introduction

PyTorch docs 中描述了 [cuda memory management](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management), 其中简单提到了 cuda 的内存分配方法，以及 PyTorch 提供的一些用于观察内存分配情况的方法，可见 [torch cuda memory](https://pytorch.org/docs/stable/torch_cuda_memory.html#torch-cuda-memory). 此外，PyTorch 允许[用户为 cuda 自定义内存分配器 (memory allocator)](https://pytorch.org/docs/stable/notes/cuda.html#using-custom-memory-allocators-for-cuda), 只需提供特定的接口即可。当然，Memory Allocator 不仅在 GPU 资源分配中至关重要，也在操作系统等其它领域不可或缺。接下来我们将动手为 cuda 实现一个内存分配器。

## Interface & First Approach

按照 PyTorch docs 的说法，我们需要用 C/C++ 实现如下两个接口：

```c++
#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
// Compile with g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC
extern "C" {
void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   cudaMalloc(&ptr, size);
   std::cout<<"alloc "<<ptr<<size<<std::endl;
   return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   std::cout<<"free "<<ptr<< " "<<stream<<std::endl;
   cudaFree(ptr);
}
}
```

然后把这两个接口 export 到 Python 中：

```python
import torch

# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    'alloc.so', 'my_malloc', 'my_free')
# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)
# This will allocate memory in the device using the new allocator
b = torch.zeros(10, device='cuda')
```

以上是一个 naive 的实现方案。在 `my_alloc()` 函数中，我们调用 `cuda_runtime_api.h` 中的 `cudaMalloc()` 函数来进行 cuda 内存分配，并在 `my_free()` 函数中调用 `cudaFree()` 函数进行内存释放。

`cudaMalloc()` 接收两个参数，它直接分配字节数为 `size` 的内存，并将首地址赋值给 `ptr` 指针，作为 `my_alloc()` 函数的返回值（暂时不用管 `device` 和 `stream` 两个参数）。如果 `cudaMalloc()` 调用失败，或者说 cuda 无法分配这段内存了，那么该函数会返回一个非零值（遵循 POSIX 规范）。`my_free()` 函数则接收一个指针 `ptr` 与字节数 `size`, 作用是释放从 `ptr` 开始、字节数为 `size` 的这段内存。

我们自然会问：这个方法有什么不好的吗？事实上，频繁调用 `cudaMalloc()` 函数会占用大量时间（类似于 C/C++ 中频繁调用 `malloc()` 一样），我们应当尽量减小 cuda 内核的调用。改进方案是：我们一次性向 cuda 申请较大的内存、置入一个“内存池”中，每当 PyTorch 需要调用 `my_alloc()` 时，就直接返回内存池中一段空闲内存的首地址。而在 PyTorch 调用 `my_free()` 时，则不直接调用 `cudaFree()`, 而是 "lazily" 把这段内存重新放回内存池中维护起来，以供之后重新分配。

## Second Approach

我们实现上述算法。

```c++
#include <sys/types.h>
#include <unordered_map>
#include <vector>
#include <assert.h>
#include <cuda_runtime.h>
#include <unistd.h>

static std::unordered_map<size_t, std::vector<void*>> free_buffers;

extern "C" {
void* managed_alloc(ssize_t byteSize, int device, cudaStream_t stream) {
  if (!byteSize)
    return 0;

  printf("allocate request: byteSize = %lld\n", (long long)byteSize);

  byteSize = ((byteSize - 1) | 1023) + 1;

  // allocate memory from free_buffers
  auto& it = free_buffers[byteSize];
  if (it.size()) {
    void *result = it.back();
    it.pop_back();
    return result;
  }

  static const size_t buffer_capacity = 5LLU * 1024 * 1024 * 1024;
  static void* __vma_buffer_addr = 0;
  static unsigned long long __vma_offset = buffer_capacity;

  assert(byteSize <= buffer_capacity); // the byteSize request should be smaller than a single buffer_capacity

  // allocate memory slice from 5GB memory buffer
  if (__vma_offset + byteSize <= buffer_capacity) {
    void *result = ((char*)__vma_buffer_addr) + __vma_offset;
    __vma_offset += byteSize;
    return result;
  }

  // allocate another new 5GB memory buffer from device memory (slow)
  static int memory_buffer_allocated_count = 0;
  printf("create a new 5GB memory buffer (%d had been allocated before)..\n", memory_buffer_allocated_count++);

  if (cudaMalloc(&__vma_buffer_addr, buffer_capacity) != 0)
    fprintf(stderr, "Runtime Error: CUDA could not allocate sufficient memory to complete the call.\n"), fflush(stdout), fflush(stderr), _exit(1);
  __vma_offset = 0;
  return managed_alloc(byteSize, device, stream);
}

void managed_free(void* ptr, ssize_t byteSize, int device, cudaStream_t stream) {
  printf("free request: size = %lld\n", (long long)byteSize);

  if (!ptr) return;
  byteSize = ((byteSize - 1) | 1023) + 1;

  // cudaFree() is slow, so just save memory block in free_buffers dict
  free_buffers[byteSize].push_back(ptr);
}
}
```

在这段代码中，我们维护了一个 `free_buffers`, 它是从 `size_t` 到 `vector<void *>` 的一个映射，`size_t` 参数代表内存池的片段长度，`vector<void *>` 参数则维护了该长度的内存池的所有空闲的指针。`managed_free()` 的最后一句 `free_buffers[byteSize].push_back(ptr);` 会把需要释放的内存首地址存进对应长度的内存池中。算法还有一个细节是，对每个输入的字节长度 `byteSize`, 我们都把它处理为 `1024` 的倍数，这样在不浪费太多资源的情况下、尽量减小了 `free_buffers` 的 `key` 的集合大小。

## Third Approach: Buddy Algorithm

然而，上述算法仍然有缺点：内存的“碎片化”现象比较严重。假设内存池中有一段长度为 `2048` 的空闲，就只能被分配给一个长度 `2048` 的内存申请——换句话说，假设此时申请一段长度 `1024` 的内存，就无法把这段 `2048` 的片段一分为二、并把前一段 `1024` 的长度分配给调用者。因此，直觉上看，我们需要支持一个更复杂的数据结构，使得可以把内存池中的一个片段进一步“分割”成若干的小片段，类似地，在 `my_free()` 中也可以检查是否有连续的若干小片段可以“合并”成一个较大的、完整的片段。这就引出了 [Buddy Algorithm](https://en.wikipedia.org/wiki/Buddy_memory_allocation).

Donald Knuth 的著作 "The Art of Computer Programming" 中详细提到了 buddy algorithm, 而 Linux kernel 也使用了改进的 buddy algorithm 来管理其内存分配。

在 buddy algorithm 中，内存池的大小、以及可分配的内存大小均为 power of two. 整个内存池为一棵二叉树的形式，根节点为最大的内存片段，每个节点均可 split 成两个孩子节点，分别代表父节点的这段内存的前后两小段（每个孩子维护的内存均为父亲的一半）。

在 `cudaMalloc()` 时，假设输入的 `byteSize` 已经转换到 power of 2, 那就检查内存树中长度为 `byteSize` 的节点是否有空闲：如果有空闲，则直接分配这一段；否则找到一段更大的空闲内存，并将其不断 split 直到 `byteSize` 的长度并分配。如果找不到更大的空闲内存，则抛出异常。

在 `cudaFree()` 时，我们把由 `ptr` 和 `byteSize` 决定的那个节点的状态置为空闲。此时检查它的 sibling（或者说 buddy, 即“伙伴”）是否同样为空闲状态，如果是，那么将它们合并 (merge), 也就是把它们的父节点的状态置为空闲。

以下是对 buddy algorithm 的（未经优化的）实现。

```c++
#include <sys/types.h>
#include <unordered_map>
#include <vector>
#include <set>
#include <assert.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <fstream>

#define MAX_CAPACITY (4LLU * 1024 * 1024 * 1024)
#define MIN_CAPACITY (1LLU * 1024)

enum node_type {
  FREE,     // free buffer
  OCCUPIED, // already allocated
  SPLIT,    // already split into two buffers
  NONEXIST  // non-exist buffer
};

class MemoryAllocator {
public:
  MemoryAllocator(size_t _capacity, void* _addr=nullptr) : addr(_addr) {
    printf("Memory Allocator initializing...\n");
    capacity = next_power_of_two(_capacity);
    assert(capacity >= MIN_CAPACITY && capacity <= MAX_CAPACITY);
    printf("capacity = %lu, MIN_CAPACITY = %lu\n", capacity, MIN_CAPACITY);
    buffer_len = (capacity / MIN_CAPACITY) * 2 - 1;
    assert(buffer_len > 0);
    printf("buffer_len = %lu\n", buffer_len);
    buffer = new uint8_t[buffer_len];
    std::fill(buffer, buffer + buffer_len, node_type::NONEXIST);
    buffer[0] = node_type::FREE;
    free_buffers[capacity].insert(0);
    printf("Memory Allocator initialized. Capacity = %lu\n", capacity);
  }
  
  ~MemoryAllocator() {
    delete buffer;
  }

  void set_addr(void* _addr) {
    addr = _addr;
  }

  void* managed_alloc(ssize_t byte_size_, bool& rc) {
    rc = false; // return code. true: success; false: failure
    if (!byte_size_) {
      printf("Error: byte size is 0\n");
      return 0;
    }
    size_t byte_size = get_corrected_byte_size(byte_size_);
    // printf("Corrected Byte Size: %lu\n", byte_size);
    if (byte_size > capacity) {
      printf("Error: byte size exceeds capacity. byte_size = %lu, capacity = %lu\n", byte_size, capacity);
      return 0;
    }
    auto& it = free_buffers[byte_size];
    if (it.size()) {
      size_t id = *it.begin();
      it.erase(it.begin());
      buffer[id] = node_type::OCCUPIED;
      rc = true;
      return (void*)((char*)addr + get_offset(id));
    }
    // find the smallest buffer that can be split into two buffers
    bool can_split = false;
    size_t buffer_size = byte_size * 2;
    while (buffer_size <= capacity) {
      if (free_buffers[buffer_size].size()) {
        can_split = true;
        break;
      }
      buffer_size *= 2;
    }
    // printf("Can split. Buffer size = %lu\n", buffer_size);
    if (!can_split) {
      rc = false; // cannot allocate memory
      printf("Error: cannot allocate memory = %lu byte\n", byte_size_);
      return 0;
    }
    // split the buffer
    while (buffer_size > byte_size) {
      size_t id = *free_buffers[buffer_size].begin();
      split(id);
      buffer_size /= 2;
    }

    assert(free_buffers[byte_size].size());
    size_t id = *free_buffers[byte_size].begin();
    free_buffers[byte_size].erase(id);
    buffer[id] = node_type::OCCUPIED;
    rc = true;
    printf("Allocated buffer size = %lu\n", byte_size);
    return (void*)((char*)addr + get_offset(id));
  }

  void managed_free(void* ptr, ssize_t byte_size_) {
    if (!ptr) return;
    size_t byte_size = get_corrected_byte_size(byte_size_);
    assert(byte_size <= capacity);
    size_t id = get_id(ptr, byte_size);
    assert(buffer[id] == node_type::OCCUPIED);
    buffer[id] = node_type::FREE;
    free_buffers[byte_size].insert(id);
    
    // after free memory, check if we can merge the buffer
    size_t parent_id = parent(id);
    while (true) {
      size_t left_id = left_child(parent_id);
      size_t right_id = right_child(parent_id);
      if (!merge(left_id, right_id)) break;
      if (parent_id == 0) break;
      parent_id = parent(parent_id);
    }

    printf("Freed buffer size = %lu\n", byte_size);
  }

private:
  std::unordered_map<size_t, std::set<size_t>> free_buffers;
  size_t capacity;
  void* addr;
  uint8_t* buffer;
  size_t buffer_len;

  inline size_t next_power_of_two(ssize_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n+1);
  }

  inline size_t get_corrected_byte_size(ssize_t byte_size) {
    size_t n = next_power_of_two(byte_size);
    if (n < MIN_CAPACITY) n = MIN_CAPACITY;
    return n;
  }

  inline size_t left_child(size_t id) {
    return 2 * id + 1;
  }

  inline size_t right_child(size_t id) {
    return 2 * id + 2;
  }

  inline size_t parent(size_t id) {
    assert(id > 0);
    return (id - 1) / 2;
  }

  inline size_t get_depth(size_t id) {
    return static_cast<size_t>(std::log2(id + 1));
  }

  inline size_t get_depth_for_buffer_size(size_t buffer_size) {
    return static_cast<size_t>(std::log2(MAX_CAPACITY / buffer_size));
  }

  size_t get_offset(size_t id) {
    size_t depth = get_depth(id);
    size_t first_id_of_depth = (1 << depth) - 1;
    size_t buffer_size = capacity / (1 << depth);
    return (id - first_id_of_depth) * buffer_size;
  }

  size_t get_id(void* ptr, size_t byte_size) {
    size_t offset = (size_t)((char*)ptr - (char*)addr);
    assert(offset % byte_size == 0);
    size_t depth = get_depth_for_buffer_size(byte_size);
    size_t first_id_of_depth = (1 << depth) - 1;
    return first_id_of_depth + (size_t)(offset / byte_size);
  }

  size_t get_buffer_size(size_t id) {
    // get_buffer_size([0]) == capacity
    // get_buffer_size([1..2]) == capacity / 2
    // get_buffer_size([3..6]) == capacity / 4
    // get_buffer_size([7..14]) == capacity / 8
    // ...
    size_t depth = get_depth(id);
    return capacity / (1 << depth);
  }

  void split(size_t id) {
    // printf("Try split buffer %lu, size = %lu\n", id, get_buffer_size(id));
    assert(buffer[id] == node_type::FREE);
    size_t id1 = left_child(id);
    size_t id2 = right_child(id);
    assert(buffer[id1] == node_type::NONEXIST && buffer[id2] == node_type::NONEXIST);
    buffer[id] = node_type::SPLIT;
    buffer[id1] = node_type::FREE;
    buffer[id2] = node_type::FREE;
    size_t parent_buffer_size = get_buffer_size(id);
    size_t buffer_size = parent_buffer_size / 2;
    assert(buffer_size >= MIN_CAPACITY);
    free_buffers[parent_buffer_size].erase(id);
    free_buffers[buffer_size].insert(id1);
    free_buffers[buffer_size].insert(id2);
  }

  bool merge(size_t id1, size_t id2) {
    if (buffer[id1] != node_type::FREE || buffer[id2] != node_type::FREE) return false;
    if (parent(id1) != parent(id2) || id1 == id2) return false;
    size_t pa = parent(id1);
    if (buffer[pa] != node_type::SPLIT) return false;

    buffer[pa] = node_type::FREE;
    size_t buffer_size = get_buffer_size(id1);
    free_buffers[buffer_size].erase(id1);
    free_buffers[buffer_size].erase(id2);
    free_buffers[buffer_size * 2].insert(pa);
    buffer[id1] = node_type::NONEXIST;
    buffer[id2] = node_type::NONEXIST;
    return true;
  }
};

#include <cuda_runtime.h>

static MemoryAllocator allocator(MAX_CAPACITY, 0);

extern "C" {
void* managed_alloc(ssize_t byte_size, int device, cudaStream_t stream) {
  if (!byte_size) return 0;

  bool rc = false;
  static bool allocated = false;
  static void* __vma_buffer_addr = 0;

  if (!allocated) {
    if (cudaMalloc(&__vma_buffer_addr, MAX_CAPACITY) != 0) {
      fprintf(stderr, "Runtime Error: CUDA could not allocate sufficient memory to complete the call.\n"), fflush(stdout), fflush(stderr), _exit(1);
    }
    allocated = true;
    allocator.set_addr(__vma_buffer_addr);
  }

  auto ptr = allocator.managed_alloc(byte_size, rc);
  if (!rc) {
    printf("Fatal error: cannot allocate memory\n");
    return 0;
  }
  return ptr;
}

void managed_free(void* ptr, ssize_t byte_size, int device, cudaStream_t stream) {
  if (!ptr) return;
  allocator.managed_free(ptr, byte_size);
  return;
}
}
```

值得注意的是，内存树并不需要显式地用树的数据结构来实现，而是可以借鉴二叉堆的实现思路，用数组来模拟。

然而，根据 Wikipedia 上的描述，这个算法仍然有碎片化的问题：例如，当我们申请 `66K` 的内存时，buddy algorithm 会直接分配 `128K` 的内存，这就导致了 `62K` 的浪费。可以使用 [Slab allocation](https://en.wikipedia.org/wiki/Slab_allocation) 来解决这个问题。许多操作系统，例如 FreeBSD 和 Linux 都使用了 Slab allocation.

