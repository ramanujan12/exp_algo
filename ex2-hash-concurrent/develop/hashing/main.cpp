#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <chrono>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <unordered_set>
#include <vector>

//constexpr int M = 1 << 27;
constexpr int M = 1 << 27;

// This function turns a hash into an index in [0, M).
// This computes h % M but since M is a power of 2,
// we can compute the modulo by using a bitwise AND to cut off the leading binary digits.
// As hash, we just use the integer key directly.
int hash_to_index(int h) {
  return h & (M - 1);
}

struct locked_linear_table {
  static constexpr const char *name = "locked-linear";
  
  struct cell {
    uint64_t key;
    uint64_t value;
    bool valid = false;
  };
  
  locked_linear_table()
    : cells{new cell[M]{}} { }
  
  // Note: to simply the implementation, destructors and copy/move constructors are missing.
  
  void put(uint64_t k, uint64_t v) {
    // Take a mutex while accessing the table.
    std::lock_guard lock{mutex};
    
    int i = 0;
    
    while(true) {
      assert(i < M);
      
      auto idx = hash_to_index(k + i);
      auto &c = cells[idx];
      
      if(!c.valid) {
	c.key = k;
	c.value = v;
	c.valid = true;
	return;
      }
      
      if(c.key == k) {
	c.value = v;
	return;
      }
      
      ++i;
    }
  }
  
  std::optional<uint64_t> get(uint64_t k) {
    // Take a mutex while accessing the table.
    std::lock_guard lock{mutex};
    
    int i = 0;
    
    while(true) {
      assert(i < M);
      
      auto idx = hash_to_index(k + i);
      auto &c = cells[idx];
      
      if(!c.valid)
	return std::nullopt;
      
      if(c.key == k)
	return c.value;
      
      ++i;
    }
  }
  
  cell *cells = nullptr;
  std::mutex mutex;
};

struct concurrent_chaining_table {
  static constexpr const char *name = "concurrent-chaining";
  
  struct chain {
    chain *next;
    uint64_t key;
    uint64_t value;
  };
  
  concurrent_chaining_table()
    : heads{new chain *[M]{}}, mutexes{new std::mutex[M]} { }
  
  // Note: to simply the implementation, destructors and copy/move constructors are missing.
  
  void put(uint64_t k, uint64_t v) {
    auto idx = hash_to_index(k);
    
    // Take a mutex while accessing the buckets.
    std::lock_guard lock{mutexes[idx]};
    
    auto p = heads[idx];
    
    if(!p) {
      heads[idx] = new chain{nullptr, k, v};
      return;
    }
    
    while(true) {
      assert(p);
      
      if(p->key == k) {
	p->value = v;
	return;
      }
      
      if(!p->next) {
	p->next = new chain{nullptr, k, v};
	return;
      }

      p = p->next;
    }
  }
  
  std::optional<uint64_t> get(uint64_t k) {
    auto idx = hash_to_index(k);
    
    // Take a mutex while accessing the buckets.
    std::lock_guard lock{mutexes[idx]};
    
    auto p = heads[idx];
    
    while(p) {
      if(p->key == k)
	return p->value;
      
      p = p->next;
    }
    
    return std::nullopt;
  }
  
  chain **heads = nullptr;
  std::mutex *mutexes = nullptr;
};

struct lockfree_linear_table {
  static constexpr const char *name = "lockfree-linear";
  static constexpr const uint64_t key_empty = uint64_t(-1);
  static constexpr const uint64_t key_tomb  = uint64_t(-2);
  
  struct alignas(16) cell {
    uint64_t key = uint64_t(-1);
    uint64_t value;
  };

  lockfree_linear_table()
    : cells{new cell[M]{}} {}
  
  // Semantics: if r matches expected, write desired to r and return true.
  //            Otherwise, read r into expected and return false.
  static bool dwcas(cell &r, cell &expected, const cell &desired) {
    // Unfortunately, GCC does not export the cmpxchg16b instruction via
    // intrinsics, so we have to roll our own using inline assembly code.
    bool res;
    // Don't worry, you do not have to understand the arcane syntax below:
    asm volatile ("lock cmpxchg16b %1"
		  : "=@ccz" (res), "+m" (r), "+a" (expected.key), "+d" (expected.value)
		  : "b" (desired.key), "c" (desired.value)
		  : "memory");
    return res;
  }
  
  void put(uint64_t k, uint64_t v) {
    int i = 0;

    cell cell_inp {k, v};
    while(true) {
      assert(i < M);
      auto idx = hash_to_index(k + i);
      
      // atomic read in key and value
      uint64_t key, val;
#pragma omp atomic read
      key = cells[idx].key;
#pragma omp atomic read
      val = cells[idx].value;

      // empty cell
      if (key == key_empty) {
	cell cell_empty {key_empty, val};
	if (dwcas(cells[idx], cell_empty, cell_inp))
	  return;
      }

      // same key -> value replace
      if (key == k) {
	cell cell_load {key, val};
	if (dwcas(cells[idx], cell_load, cell_inp))
	  return;
      }
      
      ++i;
    }
  }
  
  std::optional<uint64_t> get(uint64_t k) {
    int i = 0;

    while(true) {
      assert(i < M);
      auto idx = hash_to_index(k + i);

      // atomic read in key and value
      uint64_t key, val;
#pragma omp atomic read
      key = cells[idx].key;
#pragma omp atomic read
      val = cells[idx].value;

      // empty cell
      if (key == key_empty)
	return std::nullopt;

      if (key == k)
	return val;

      ++i;
    }
  }

  cell *cells = nullptr;
};

template<typename Algo>
void evaluate() {
  Algo table;
  
  unsigned int n = M * 0.9;
  
  std::mt19937 global_prng{42};
  
  std::cerr << "Generating random data..." << std::endl;
  
  // Now generate a sequence of insertions that only use keys in our pool.
  std::vector<std::pair<uint64_t, uint64_t>> inserts;
  
  {
    std::uniform_int_distribution<uint64_t> distrib{0, (uint64_t{1} << 63) - 1};
    std::unordered_set<uint64_t> temp;
    while(inserts.size() < n) {
      uint64_t k = distrib(global_prng);
      if(temp.find(k) != temp.end())
	continue;
      inserts.push_back({k, inserts.size()});
      temp.insert(k);
    }
  }
  
  // Fill the table with junk values to test lookup later.
  std::cerr << "Prefilling table..." << std::endl;
  
#pragma omp parallel
  {
    std::mt19937 local_prng{21u + omp_get_thread_num()};
    std::uniform_int_distribution<uint64_t> distrib{0, (uint64_t{1} << 63) - 1};
    
#pragma omp for
    for(size_t i = 0; i < inserts.size() / 3; ++i) {
      auto [k, v] = inserts[i];
      (void)v;
      table.put(k, distrib(local_prng));
    }
  }

  // We want iterate through the keys in random order to avoid any potential bias.
  std::shuffle(inserts.begin(), inserts.end(), global_prng);
  
  std::cerr << "Performing insertions..." << std::endl;
  
  auto insert_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
  for(size_t i = 0; i < inserts.size(); ++i) {
    auto [k, v] = inserts[i];
    table.put(k, v);
  }
  auto t_insert = std::chrono::high_resolution_clock::now() - insert_start;
  
  std::cerr << "Performing lookups..." << std::endl;
  
  // We want iterate through the keys in random order to avoid any potential bias.
  std::shuffle(inserts.begin(), inserts.end(), global_prng);
  
  int errors = 0;
  auto lookup_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
  for(size_t i = 0; i < inserts.size(); ++i) {
    auto [k, v] = inserts[i];
    auto r = table.get(k);
    if(!r) {
      ++errors;
      continue;
    }
    
    if(*r != v) {
      ++errors;
      continue;
    }
  }
  auto t_lookup = std::chrono::high_resolution_clock::now() - lookup_start;
  
  std::cerr << "There were " << errors << " errors" << std::endl;
  assert(!errors);
  
  std::cout << "threads: " << omp_get_max_threads() << std::endl;
  std::cout << "algo: " << Algo::name << std::endl;
  std::cout << "m: " << M << std::endl;
  std::cout << "time_insert: "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t_insert).count()
	    << " # ms" << std::endl;
  std::cout << "time_lookup: "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t_lookup).count()
	    << " # ms" << std::endl;
}

static const char *usage_text =
			  "Usage: hashing [OPTIONS]\n"
			  "Possible OPTIONS are:\n"
			  "    --algo ALGORITHM\n"
			  "        Select an algorithm {chaining,linear,stl}.";

int main(int argc, char **argv) {
  std::string_view algorithm;
  
  auto error = [] (const char *text) {
    std::cerr << usage_text << "Usage error: " << text << std::endl;
    exit(2);
  };
  
  // Argument for unary options.
  const char *arg;
  
  // Parse all options here.
  
  char **p = argv + 1;
  
  auto handle_unary_option = [&] (const char *name) -> bool {
    assert(*p);
    if(std::strcmp(*p, name))
      return false;
    ++p;
    if(!(*p))
      error("expected argument for unary option");
    arg = *p;
    ++p;
    return true;
  };
  
  while(*p && !std::strncmp(*p, "--", 2)) {
    if(handle_unary_option("--algo")) {
      algorithm = arg;
    }else{
      error("unknown command line option");
    }
  }
  
  if(*p)
    error("unexpected arguments");
  
  // Verify that options are correct and run the algorithm.
  
  if(algorithm.empty())
    error("no algorithm specified");
  
  if(algorithm == "locked-linear") {
    evaluate<locked_linear_table>();
  }else if(algorithm == "concurrent-chaining") {
    evaluate<concurrent_chaining_table>();
  }else if(algorithm == "lockfree-linear") {
    evaluate<lockfree_linear_table>();
  }else{
    error("unknown algorithm");
  }
}
