#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/Registry.h>
#include <c10/util/SmallVector.h>
#include <mutex>
#include <atomic>

#include "csrc/runtime/raw_mem_manager.h"
#include "csrc/runtime/Runtime.h"

using namespace c10;

namespace oplifter {

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};
// A 4*3 array, basicly.
typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from cudaMalloc().
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via cudaFree)
  StatArray inactive_split;

  // SUM: bytes allocated by this memory alocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to CUDA malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to CUDA after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // COUNT: total number of synchronize_and_free_events() calls
  int64_t num_sync_all_streams = 0;

  // COUNT: total number of CUDA allocation calls. This includes both cuMemMap
  // and cudaMalloc.
  int64_t num_device_alloc = 0;

  // COUNT: total number of CUDA free calls. This includes both cuMemUnmap
  // and cudaFree.
  int64_t num_device_free = 0;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

struct Block;

class OliDeviceCachingAllocator;

#define kNumMutexShard 67

class OliCachingAllocator final : public at::Allocator {
private:
  // Shard allocation region to have independent mutexes to reduce contention.
  // static constexpr size_t kNumMutexShard = 67;
  // TODO: use std::hardware_destructive_interference_size once available
  struct alignas(64) AlignedMutex {
    std::mutex m;
  };
  static std::array<AlignedMutex, kNumMutexShard> mutex;
  // allocated blocks by device pointer
  static std::array<ska::flat_hash_map<void*, Block*>, kNumMutexShard>
      allocated_blocks;

  static std::array<size_t, 3> caching_recoder;
public:
  OliCachingAllocator() = default;
  
  static void uncached_delete(void* ptr) {
    oplifter::free_raw_mem(ptr);
  }

  at::DataPtr allocate(size_t nbytes) override;

  static void local_raw_delete(void* ptr);

  at::DeleterFnPtr raw_deleter() const override {
    if (getenv("PYTORCH_NO_OLI_MEMORY_CACHING") != nullptr) {
      return &uncached_delete;
    } else {
      return &(this->local_raw_delete);
    }
  }

  // Requires: src and dest were allocated by this allocator
  // Requires: src and dest both have length >= count
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    printf("copy_data not here\n");
  }

  static std::vector<std::unique_ptr<OliDeviceCachingAllocator>> device_allocator;
  
  static Block* get_allocated_block(void* ptr, bool remove = false);

  void init(int device_count);
};

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

typedef std::shared_ptr<GatheredContext> (*CreateContextFn)();

} // oplifter

