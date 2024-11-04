#include <c10/util/CallOnce.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>

#include <c10/util/Exception.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <utility>
#include <vector>

#include "csrc/runtime/OplifterCachingAllocator.h"

// Included here as this is externally used in CUDAAllocatorConfig
const size_t kLargeBuffer = 20971520;
// all sizes are rounded to at least 512 bytes
constexpr size_t kMinBlockSize = 512; 
// largest "small" allocation is 1 MiB
constexpr size_t kSmallSize = 1048576; 
// "small" allocations are packed in 2 MiB blocks
constexpr size_t kSmallBuffer = 2097152; 
 // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kMinLargeAlloc = 10485760;
constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB

// using namespace oplifter;

namespace oplifter {

// Size pretty-printer
std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

typedef bool (*Comparison)(const Block*, const Block*);
static bool BlockComparatorSize(const Block* a, const Block* b);

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void increase_stat(Stat& stat, size_t amount) {
  stat.current += static_cast<int64_t>(amount);
  stat.peak = std::max(stat.current, stat.peak);
  stat.allocated += static_cast<int64_t>(amount);
}

void decrease_stat(Stat& stat, size_t amount) {
  stat.current -= static_cast<int64_t>(amount);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stat.current >= 0,
      "Negative tracked stat in OPL allocator (likely logic error).");
  stat.freed += static_cast<int64_t>(amount);
}

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void decrease_stat_array(StatArray& stat_array, size_t amount, const StatTypes& stat_types) {
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        decrease_stat(stat_array[stat_type], amount);
      });
}

static void uncached_delete(void* ptr) {
  oplifter::free_raw_mem(ptr);
}

struct BlockPool {
  BlockPool(bool small)
      : blocks(BlockComparatorSize),
        is_small(small) {}

  // Do not insert a Block to blocks directly; use insert_into_blocks(),
  // instead.
  std::set<Block*, Comparison> blocks;
  const bool is_small;
  int64_t get_free_blocks_call_count{0};

  // Add a Block into blocks set with updating gc counter.
  std::pair<std::set<Block*, Comparison>::iterator, bool> insert_into_blocks(Block* block);
};

struct Block {
  c10::DeviceIndex device; // gpu
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  BlockPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  bool mapped{true}; // is the virtual address range this Block references
                     // backed by physical pages. Always true when
                     // expandable_segment_ is null. When false
                     // This Block will be aligned to the segment size
                     // of its expandable_segment_.
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  // int event_count{0}; // number of outstanding CUDA events
  int64_t gc_count_base{0}; // get_free_blocks_call_count when Block is inserted
  std::shared_ptr<GatheredContext> context_when_allocated;
  // only set for the first block in the segment (when prev == null)
  // this records the frame information when malloc was called
  // whereas context_when_allocated records the last time we handed this
  // memory out from our cache.
  std::shared_ptr<GatheredContext> context_when_segment_allocated;

  // ExpandableSegment* expandable_segment_{nullptr};

  Block(
      c10::DeviceIndex device,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr),
        gc_count_base(0) {}

  // constructor for search key
  Block(c10::DeviceIndex device, size_t size)
      : device(device),
        size(size),
        requested_size(0) {}

  size_t gc_count() {
    TORCH_INTERNAL_ASSERT(pool);
    return static_cast<int>(pool->get_free_blocks_call_count - gc_count_base);
  }

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
  void splice(Block* before, Block* after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }
};

static bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

std::pair<std::set<Block*, Comparison>::iterator, bool> BlockPool::insert_into_blocks(
    Block* block){
  block->gc_count_base = get_free_blocks_call_count;
  return blocks.insert(block);
}

struct AllocParams {
  AllocParams(
      c10::DeviceIndex device,
      size_t size,
      BlockPool* pool,
      size_t alloc_size,
      DeviceStats& stats)
      : search_key(device, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr) {}

  c10::DeviceIndex device() const {
    return search_key.device;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {false};
};

class CachingAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() {
    return instance().m_expandable_segments;
  }

  static CachingAllocatorConfig &instance() {
    static CachingAllocatorConfig *s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      const char* env = getenv("PYTORCH_OPL_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:
  size_t m_max_split_size;
  double m_garbage_collection_threshold;
  bool m_expandable_segments;
  bool set_expandable_segments_flag = false;

  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_garbage_collection_threshold(0),
        m_expandable_segments(false)
        {
          // TODO: expandable_segments
          m_expandable_segments = false;
        }

  void lexArgs(const char* env, std::vector<std::string>& config) {
    std::vector<char> buf;

    size_t env_length = strlen(env);
    for (size_t i = 0; i < env_length; i++) {
      if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
        if (!buf.empty()) {
          config.emplace_back(buf.begin(), buf.end());
          buf.clear();
        }
        config.emplace_back(1, env[i]);
      } else if (env[i] != ' ') {
        buf.emplace_back(static_cast<char>(env[i]));
      }
    }
    if (!buf.empty()) {
      config.emplace_back(buf.begin(), buf.end());
    }
  }

  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c) {
    TORCH_CHECK(
        i < config.size() && config[i].compare(std::string(1, c)) == 0,
        "Error parsing CachingAllocator settings, expected ", c);
  }

  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
      size_t val1 = static_cast<size_t>(stoi(config[i]));
      TORCH_CHECK(
          val1 > kLargeBuffer / (1024 * 1024),
          "CachingAllocator option max_split_size_mb too small, must be > ",
          kLargeBuffer / (1024 * 1024));
      val1 = std::max(val1, kLargeBuffer / (1024 * 1024));
      val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
      m_max_split_size = val1 * 1024 * 1024;
    } else {
      TORCH_CHECK(false, "Error, expecting max_split_size_mb value");
    }
    return i;
  }

  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
      double val1 = stod(config[i]);
      TORCH_CHECK(
          val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0");
      TORCH_CHECK(
          val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0");
      m_garbage_collection_threshold = val1;
    } else {
      TORCH_CHECK(
          false, "Error, expecting garbage_collection_threshold value");
    }
    return i;
  }

  size_t parseExpandableSegments(
      const std::vector<std::string>& config,
      size_t i) {
    consumeToken(config, ++i, ':');
    if (++i < config.size()) {
      TORCH_CHECK(
          i < config.size() && (config[i] == "True" || config[i] == "False"),
          "Expected a single True/False argument for expandable_segments");
      m_expandable_segments = (config[i] == "True");
      if (m_expandable_segments) {
        // TODO
      }
    } else {
      TORCH_CHECK(false, "Error, expecting expandable_segments value");
    }
    return i;
  }
};

void CachingAllocatorConfig::parseArgs(const char* env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_garbage_collection_threshold = 0;

  if (env == nullptr) {
    return;
  }

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    if (config[i].compare("max_split_size_mb") == 0) {
      i = parseMaxSplitSize(config, i);
    } else if (config[i].compare("garbage_collection_threshold") == 0) {
      i = parseGarbageCollectionThreshold(config, i);
    } else if (config[i] == "expandable_segments") {
      set_expandable_segments_flag = true;
      i = parseExpandableSegments(config, i);
    } else {
      TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i]);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }

  if (m_expandable_segments) {
      if (set_expandable_segments_flag) {
          TORCH_CHECK(m_max_split_size == std::numeric_limits<size_t>::max() && m_garbage_collection_threshold == 0,
                      "`max_split_size_mb` or `garbage_collection_threshold`, cannot be enabled with "
                      "`expandable_segments`, please set `expandable_segments` to `false`.");
      } else if (m_max_split_size != std::numeric_limits<size_t>::max() || m_garbage_collection_threshold != 0) {
          m_expandable_segments = false;
          printf("`max_split_size_mb` or `garbage_collection_threshold` is enabled, and the "
                              "`expandable_segments` is changed to `false` by default.\n");
      }
  }
}

class OliDeviceCachingAllocator {
private:
  // lock around all operations
  mutable std::recursive_mutex mutex;
  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;
  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  ska::flat_hash_set<Block*> active_blocks;

  // record used memory.
  size_t total_allocated_memory = 0;
  size_t allowed_memory_maximum = 0;

  bool set_fraction = false;

  std::atomic<CreateContextFn> context_recorder_;
  RecordContext record_context_ = RecordContext::NEVER;

public:
  OliDeviceCachingAllocator()
      : large_blocks(/*small=*/false),
        small_blocks(/*small=*/true) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
    context_recorder_.store(nullptr);
  }

  // Must be called outside of `mutex` or deadlocks are possible with Python
  std::shared_ptr<GatheredContext> maybeGatherContext(RecordContext level) {
    return nullptr;
  }

  Block* malloc(c10::DeviceIndex device, size_t orig_size) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    auto context = maybeGatherContext(RecordContext::STATE);

    std::unique_lock<std::recursive_mutex> lock(mutex);

    size_t size = round_size(orig_size);
    auto& pool = size <= kSmallSize ? small_blocks : large_blocks;
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, &pool, alloc_size, stats);
    params.stat_types = get_stat_types_for_pool(pool);

    // First, try to get a block from the existing pool.
    bool block_found = get_free_block(params);

    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(
              set_fraction &&
              CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_blocks(context);
      }
      // Attempt allocate
      // WARNING: alloc_block may release the allocator lock when calling
      // cudaMalloc. So far this function has not modified allocator state, but
      // keep in mind that any observed allocator state may change across calls
      // to alloc_block since it may release the lock.
      block_found = alloc_block(params, false, context, lock)
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          || (release_available_cached_blocks(params, context) &&
              alloc_block(params, false, context, lock))
          // Free all non-split cached blocks and retry alloc.
          || (release_cached_blocks(context) &&
              alloc_block(params, true, context, lock));
    }

    if (!block_found) {
      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "Out of memory. Tried to allocate ",
          format_size(alloc_size),
          ", Failed.");
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(
        std::move(params), orig_size, std::move(context), split_remainder);
  }

  void free(Block* block) {
    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      decrease_stat(stats.allocation[stat_type], 1);
      decrease_stat(stats.allocated_bytes[stat_type], block->size);
    });

    if (block->size >= CachingAllocatorConfig::max_split_size())
      decrease_stat(stats.oversize_allocations, 1);

    free_block(block, context);
    // c10::reportMemoryUsageToProfiler();
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(!block->allocated);

    block->context_when_allocated = nullptr;
    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.insert_into_blocks(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types = get_stat_types_for_pool(pool);

    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // inactive_split tries to capture the idea that blocks
      // cannot be freed when requested, but fully free pages
      // of expandable blocks can always be freed.
      // The logic to track this as statistic is pretty involved,
      // so we simply just exclude expandable segments from
      // inactive_split
      if (net_change_inactive_split_blocks > 0) {
        increase_stat(
            stats.inactive_split[stat_type],
            static_cast<size_t>(net_change_inactive_split_blocks));
      } else if (net_change_inactive_split_blocks < 0) {
        decrease_stat(
            stats.inactive_split[stat_type],
            static_cast<size_t>(-net_change_inactive_split_blocks));
      }
      if (net_change_inactive_split_size > 0) {
        increase_stat(
            stats.inactive_split_bytes[stat_type],
            static_cast<size_t>(net_change_inactive_split_size));
      } else if (net_change_inactive_split_size < 0) {
        decrease_stat(
            stats.inactive_split_bytes[stat_type],
            static_cast<size_t>(-net_change_inactive_split_size));
      }
      
      decrease_stat(stats.active[stat_type], 1);
      decrease_stat(stats.active_bytes[stat_type], original_block_size);
      decrease_stat(stats.requested_bytes[stat_type], requested_size);
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
      dst->context_when_segment_allocated =
          std::move(src->context_when_segment_allocated);
    } else { // [dest src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = pool.blocks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  Block* alloc_found_block(AllocParams params, size_t orig_size, 
      std::shared_ptr<GatheredContext> context, bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;

    TORCH_INTERNAL_ASSERT(params.block != nullptr &&
        params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (split_remainder) {
      remaining = block;

      block = new Block(device, size, pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      bool inserted = pool->insert_into_blocks(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

      if (already_split ) {
        // An already-split inactive block is being shrunk by size bytes.
        decrease_stat_array(
            stats.inactive_split_bytes, block->size, params.stat_types);
      } else {
        // A new split inactive block is being created from a previously unsplit
        // block, size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          increase_stat(stats.inactive_split_bytes[stat_type], remaining->size);
          increase_stat(stats.inactive_split[stat_type], 1);
        });
      }

    } else if (already_split) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        decrease_stat(stats.inactive_split_bytes[stat_type], block->size);
        decrease_stat(stats.inactive_split[stat_type], 1);
      });
    }

    block->allocated = true;
    block->requested_size = orig_size;

    block->context_when_allocated = std::move(context);

    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      increase_stat(stats.allocation[stat_type], 1);
      increase_stat(stats.allocated_bytes[stat_type], block->size);
      increase_stat(stats.active[stat_type], 1);
      increase_stat(stats.active_bytes[stat_type], block->size);
      increase_stat(stats.requested_bytes[stat_type], block->requested_size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      increase_stat(stats.oversize_allocations, 1);

    // c10::reportMemoryUsageToProfiler(
    //     block->ptr,
    //     block->size,
    //     stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
    //     stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
    //     c10::Device(c10::DeviceType::CUDA, device));

    return block;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }

  // This function assumes that global lock has been taken whle calling into
  // this function. We do cudaMalloc sync call in this function which
  // can be expensive while holding the lock. Hence, we pass-in the lock to the
  // function to temporarily release the lock before cudaMalloc call and acquire
  // it back again after the call so that other threads dont get blocked.
  bool alloc_block(AllocParams& p, bool isRetry,
      const std::shared_ptr<GatheredContext>& ctx,
      std::unique_lock<std::recursive_mutex>& lock) {

    size_t size = p.alloc_size;
    void* ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      // p.err = cudaErrorMemoryAllocation;
      return false;
    } else {
      ptr = oplifter::alloc_raw_mem(size);
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      increase_stat(stats.segment[stat_type], 1);
      increase_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
      increase_stat(stats.oversize_segments, 1);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    stats.num_device_alloc++;
    p.block->context_when_segment_allocated = ctx;
    return true;
  }

  void release_block(Block* block, const std::shared_ptr<GatheredContext>& context) {
    stats.num_device_free++;

    oplifter::free_raw_mem((void*)block->ptr);
    total_allocated_memory -= block->size;

    auto* pool = block->pool;

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      decrease_stat(stats.segment[stat_type], 1);
      decrease_stat(stats.reserved_bytes[stat_type], block->size);
    });

    if (block->size >= CachingAllocatorConfig::max_split_size())
      decrease_stat(stats.oversize_segments, 1);
    pool->blocks.erase(block);
    delete block;
  }

  // Free one or more oversize blocks to the system allocator.  
  // But only enough to satisfy the target size
  bool release_available_cached_blocks(const AllocParams& p,
      const std::shared_ptr<GatheredContext>& context) {
    if (CachingAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    BlockPool& pool = *p.pool;

    // because of std::unique_ptr, block cannot be trivially copied
    // Use constructor for search key.
    Block key(p.search_key.device, p.search_key.size);
    key.size = (key.size < CachingAllocatorConfig::max_split_size())
        ? CachingAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;
      size_t totalReleased = 0;
      --it; // Back up one item.  Now on the largest block for the correct.
      while ((totalReleased < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur, context);
        } else {
          release_block(*cur, context);
          break;
        }
      }
      if (totalReleased < key.size)
        return false;
    } else {
      release_block(*it, context);
    }
    return true;
  }

  bool release_cached_blocks(const std::shared_ptr<GatheredContext>& context) {
    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks, context);
    release_blocks(small_blocks, context);

    return true;
  }

  void release_blocks(BlockPool& pool, const std::shared_ptr<GatheredContext>& context) {
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block, context);
      }
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;

    // if 'memory fraction' is used and it is reached, try to call block_free.
    if (C10_UNLIKELY(
            set_fraction &&
            CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      ++ pool.get_free_blocks_call_count;
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end())
      return false;

    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }
  
  void garbage_collect_cached_blocks(
      const std::shared_ptr<GatheredContext>& context) {
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        CachingAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto& b : large_blocks.blocks) {
      if (!b->is_split()) {
        total_age += b->gc_count();
        ++freeable_block_count;
      }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true &&
           freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        Block* block = *it;
        ++it;
        if (!block->is_split() && block->gc_count() >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count(); // Decrement the age
          freeable_block_count--; // One less block that can be freed
          release_block(block, context);
        }
      }
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      // TODO: use roundup_power2_divisions
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }
  
  StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }
};  // OliDeviceCachingAllocator

std::array<OliCachingAllocator::AlignedMutex, kNumMutexShard> OliCachingAllocator::mutex;

std::array<ska::flat_hash_map<void*, Block*>, kNumMutexShard>
    OliCachingAllocator::allocated_blocks;

std::vector<std::unique_ptr<OliDeviceCachingAllocator>> OliCachingAllocator::device_allocator;

at::DataPtr OliCachingAllocator::allocate(size_t nbytes) {
  constexpr size_t one_exa_bytes = 1152921504606846976ULL;
  TORCH_CHECK_WITH(
      OutOfMemoryError,
      nbytes < one_exa_bytes,
      "CUDA out of memory. Tried to allocate more than 1EB memory.");
  
  c10::DeviceIndex device = oplifter::dpcppGetCurDevice();
  void* devPtr = nullptr;
  void (*deleteFunc)(void*) = &local_raw_delete;
  // dnnl::stream stream = GpuStreamManager::Instance().get_stream();

  bool force_uncached = getenv("PYTORCH_NO_OPL_MEMORY_CACHING") != nullptr;
  if (force_uncached) {
    deleteFunc = &uncached_delete;
    devPtr = oplifter::alloc_raw_mem(nbytes);
  } else {
    if (nbytes != 0) {
      if(device_allocator.size() == 0 ||
          static_cast<size_t>(device) < device_allocator.size()) {
        int dev_count = oplifter::dpcppGetDeviceCount();
        TORCH_CHECK(0 < dev_count, "No available device for Allocator to initialize.")
        init(dev_count);
      }
      Block* block = device_allocator[device]->malloc(device, nbytes);
      // add_allocated_block;
      const auto mutex_shard_id = twang_mix64((size_t)block->ptr) % kNumMutexShard;
      std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
      allocated_blocks[mutex_shard_id][block->ptr] = block;
      devPtr = (void*)block->ptr;
    }
  }
  
  // printf("allocate:%p\n", devPtr);
  return {devPtr, devPtr, deleteFunc, Device(DeviceType::PrivateUse1, device)};
}

void OliCachingAllocator::local_raw_delete(void* ptr) {
  // printf("delete:%p\n", ptr);

  if (!ptr) {
    return;
  }
  Block* block = get_allocated_block(ptr, true /* remove */);
  if (!block) {
    TORCH_CHECK(false, "invalid device pointer: ", ptr);
  }
  device_allocator[block->device]->free(block);
}

Block* OliCachingAllocator::get_allocated_block(void* ptr, bool remove) {
  // get_mutex_shard_id
  const auto mutex_shard_id = twang_mix64((size_t)ptr) % kNumMutexShard;
  // 
  std::lock_guard<std::mutex> lock(mutex[mutex_shard_id].m);
  auto it = allocated_blocks[mutex_shard_id].find(ptr);
  if (it == allocated_blocks[mutex_shard_id].end()) {
    return nullptr;
  }
  Block* block = it->second;
  if (remove) {
    allocated_blocks[mutex_shard_id].erase(it);
  }
  return block;
}

void OliCachingAllocator::init(int device_count) {
  int size = static_cast<int>(device_allocator.size());
  if (size < device_count) {
    device_allocator.resize(device_count);
    for(int i = size; i<device_count; i++){
      device_allocator[i] = std::make_unique<OliDeviceCachingAllocator>();
    }
  }
}

} // oplifter