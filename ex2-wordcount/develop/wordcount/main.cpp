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
#include <vector>

#include <bitset>
#include <unordered_set>

constexpr int M = 1u << 26u;

// This function turns a hash into an index in [0, M).
// This computes h % M but since M is a power of 2,
// we can compute the modulo by using a bitwise AND to cut off the leading binary digits.
// As hash, we just use the integer key directly.
int hash_to_index(int h) {
    return h & (M - 1);
}

// Note: this table only implements put() and no values.
struct concurrent_chaining_table {
    struct chain {
        chain *next;
        unsigned int key;
    };

    chain **heads = nullptr;
    std::mutex *mutexes = nullptr;
    int count;
    std::mutex count_mutex;

    concurrent_chaining_table()
        : heads{new chain *[M]{}}, mutexes{new std::mutex[M]}, count{0} { }

    // Note: to simplify the implementation, destructors and copy/move constructors are missing.

    void put(unsigned int k) {
        auto idx = hash_to_index(k);

        // Take a mutex while accessing the buckets.
        std::lock_guard lock{mutexes[idx]};

        auto p = heads[idx];

        if(!p) {
            heads[idx] = new chain{nullptr, k};
            // Create scope block for count mutex
            {
                std::lock_guard<std::mutex> count_lock{count_mutex};
                ++count;
            }
            return;
        }

        while(true) {
            assert(p);

            if(p->key == k)
                return;

            if(!p->next) {
                p->next = new chain{nullptr, k};
                // Create scope block for count mutex
                {
                    std::lock_guard<std::mutex> count_lock{count_mutex};
                    ++count;
                }
                return;
            }

            p = p->next;
        }
    }

    int get_count() {
        return this->count;
    }
};

size_t count_substrings(const std::vector<uint8_t> &input) {

    concurrent_chaining_table chaining_table {};

    for (size_t i = 0; i < (input.size() - 3u); ++i) {
        unsigned int key =
                  (static_cast<unsigned int>(input[i]) << 24u)
                | (static_cast<unsigned int>(input[i + 1]) << 16u)
                | (static_cast<unsigned int>(input[i + 2]) << 8u)
                | static_cast<unsigned int>(input[i + 3]);

        chaining_table.put(key);
    }
    return chaining_table.get_count();
}

int main() {
    std::vector<uint8_t> input;
    input.resize(1 << 28);

    std::mt19937 prng{42};

    std::cerr << "Generating random data..." << std::endl;

    std::uniform_int_distribution<int> distrib{0, 255};
    for(size_t i = 0; i < input.size(); ++i)
                input[i] = distrib(prng);

    std::cerr << "Running benchmark..." << std::endl;

    auto bench_start = std::chrono::high_resolution_clock::now();
    auto result = count_substrings(input);
    auto bench_elapsed = std::chrono::high_resolution_clock::now() - bench_start;

    std::cout << "num_threads: " << omp_get_max_threads() << std::endl;
    std::cout << "result: " << result << std::endl;
    std::cout << "time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(bench_elapsed).count()
              << " # ms" << std::endl;
}
