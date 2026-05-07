#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <cstring>            // for strerror
#include <mutex>
#include <unordered_map>
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
#include "mem_alloc.h"

// Track allocations that used the mmap fallback path (with or without
// cudaHostRegister) so free_pinned_ptr can choose the correct deallocation.
static std::mutex g_fallback_mtx;
struct FallbackInfo {
  size_t size;
  bool pinned;  // true if cudaHostRegister succeeded
};
static std::unordered_map<uintptr_t, FallbackInfo> g_fallback_allocs;

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, size, flags);
  if (err == cudaSuccess) {
    return reinterpret_cast<uintptr_t>(ptr);
  }

  // cudaHostAlloc can fail on certain GPU architectures (e.g. Blackwell)
  // or driver configurations.  Clear the error state and fall back to
  // mmap + cudaHostRegister, which produces equivalent pinned memory.
  (void)cudaGetLastError();

  ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("alloc_pinned_ptr mmap failed: ") +
                             strerror(errno));
  }

  // Force physical page allocation so cudaHostRegister can pin them.
  const long ps = sysconf(_SC_PAGESIZE);
  for (size_t off = 0; off < size; off += ps) {
    volatile char* c = static_cast<volatile char*>(ptr) + off;
    *c = 0;
  }

  bool pinned = false;
  err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
  if (err == cudaSuccess) {
    pinned = true;
  } else {
    // cudaHostRegister failed.  CUDA kernels (get_kernel_ptr) require
    // pinned memory to obtain a device-accessible pointer via
    // cudaHostGetDevicePointer.  Continuing with unpinned memory would
    // cause illegal memory accesses later.  Release the mapping and
    // propagate the error.
    (void)cudaGetLastError();
    std::string msg =
        std::string("alloc_pinned_ptr: cudaHostRegister failed (") +
        cudaGetErrorString(err) +
        "). Unable to pin host memory — CUDA kernels require pinned "
        "memory for host-device transfers. This can happen on certain "
        "GPU architectures (e.g. Blackwell) or when the pinned-memory "
        "limit is exceeded. Try reducing max_local_cpu_size or check "
        "driver configuration.";
    munmap(ptr, size);
    throw std::runtime_error(msg);
  }

  uintptr_t result = reinterpret_cast<uintptr_t>(ptr);
  {
    std::lock_guard<std::mutex> lk(g_fallback_mtx);
    g_fallback_allocs[result] = {size, pinned};
  }
  return result;
}

void free_pinned_ptr(uintptr_t ptr) {
  void* p = reinterpret_cast<void*>(ptr);

  // Check if this allocation used the mmap fallback path.
  FallbackInfo info = {0, false};
  {
    std::lock_guard<std::mutex> lk(g_fallback_mtx);
    auto it = g_fallback_allocs.find(ptr);
    if (it != g_fallback_allocs.end()) {
      info = it->second;
      g_fallback_allocs.erase(it);
    }
  }

  if (info.size > 0) {
    // Fallback path: unregister if pinned, then release the mapping.
    if (info.pinned) {
      cudaError_t err = cudaHostUnregister(p);
      if (err != cudaSuccess) {
        (void)cudaGetLastError();
      }
    }
    if (munmap(p, info.size) != 0) {
      throw std::runtime_error(
          std::string("free_pinned_ptr munmap failed: ") + strerror(errno));
    }
  } else {
    // Standard path: memory was allocated with cudaHostAlloc.
    cudaError_t err = cudaFreeHost(p);
    if (err != cudaSuccess) {
      (void)cudaGetLastError();
      throw std::runtime_error(
          std::string("cudaFreeHost failed: ") + cudaGetErrorString(err));
    }
  }
}

static void first_touch(void* p, size_t size) {
  const long ps = sysconf(_SC_PAGESIZE);
  for (size_t off = 0; off < size; off += ps) {
    volatile char* c = (volatile char*)p + off;
    *c = 0;
  }
}

static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

uintptr_t alloc_numa_ptr(size_t size, int node) {
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));

  // Maximum of 64 numa nodes
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  if (mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  first_touch(ptr, size);

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  void* ptr = reinterpret_cast<void*>(alloc_numa_ptr(size, node));

  cudaError_t st = cudaHostRegister(ptr, size, 0);
  if (st != cudaSuccess) {
    (void)cudaGetLastError();
    munmap(ptr, size);
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  // Unpin first, then unmap.
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    (void)cudaGetLastError();
    munmap(p, size);
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

uintptr_t alloc_shm_pinned_ptr(size_t size, const std::string& shm_name) {
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0)
    throw std::runtime_error(std::string("shm_open failed: ") +
                             strerror(errno));

  if (ftruncate(fd, size) != 0) {
    int err = errno;
    close(fd);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("ftruncate failed: ") + strerror(err));
  }

  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
  }

  first_touch(ptr, size);

  cudaError_t st = cudaHostRegister(ptr, size, 0);
  if (st != cudaSuccess) {
    (void)cudaGetLastError();
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_shm_pinned_ptr(uintptr_t ptr, size_t size,
                         const std::string& shm_name) {
  void* p = reinterpret_cast<void*>(ptr);
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    (void)cudaGetLastError();
    munmap(p, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
  shm_unlink(shm_name.c_str());
}
