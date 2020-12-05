#include <algorithm>
#include <cassert>
#include <cstring>
#include <chrono>
#include <iostream>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include <unistd.h>

int num_insertions = 1u << 26u;

// This hash table uses linear probing.
struct static_table {
	static constexpr const char *name = "static";

	struct cell {
		int key;
		int value;
		bool valid = false;
	};

	static_table(float max_fill)
	: cells{new cell[m]{}} {
		(void)max_fill; // Ignore the fill factor.
	}

	// Note: to simply the implementation, destructors and copy/move constructors are missing.

	// This function turns a hash into an index in [0, m).
	// This computes h % m but since m is a power of 2,
	// we can compute the modulo by using a bitwise AND to cut off the leading binary digits.
	// As hash, we just use the integer key directly.
	int hash_to_index(int h) {
		return h & (m - 1);
	}

	void put(int k, int v) {
		int i = 0;

		while(true) {
			assert(i < m);

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

	std::optional<int> get(int k) {
		int i = 0;

		while(true) {
			assert(i < m);

			auto idx = hash_to_index(k + i);
			auto &c = cells[idx];

			if(!c.valid)
				return std::nullopt;

			if(c.key == k)
				return c.value;

			++i;
		}
	}

	int m = 2 * num_insertions;
	cell *cells = nullptr;
};

// This hash table uses linear probing, uses the modulo strategy to determine a cell and resizes dynamicly
struct dynamic_modulo_table {
	static constexpr const char *name = "dynamic_modulo";

	struct cell {
		int key;
		int value;
		bool valid = false;
	};

    size_t table_size = 8;
    size_t insertions = 0;
	cell *cells = nullptr;
    float max_fill;

    // Initidalize our array with 8 cells
    dynamic_modulo_table(float max_fill) : cells{new cell[table_size]{}} {
        this->max_fill = max_fill;
    }

	// Note: to simply the implementation, destructors and copy/move constructors are missing.

	// This function turns a hash into an index in [0, m).
	// This computes h % m but since m is a power of 2,
	// we can compute the modulo by using a bitwise AND to cut off the leading binary digits.
	// As hash, we just use the integer key directly.
	int hash_to_index(int h) {
		return h & (table_size - 1);
	}

	void put(int k, int v) {
		int i = 0;

		while(true) {
			assert(i < table_size);

			auto idx = hash_to_index(k + i);
			auto &c = cells[idx];

			if(!c.valid) {
				c.key = k;
				c.value = v;
				c.valid = true;
                ++insertions;
                if((float)insertions / table_size > max_fill) {
                    resize_table();
                }
				return;
			}

			if(c.key == k) {
				c.value = v;
				return;
			}

			++i;
		}
	}

	std::optional<int> get(int k) {
		int i = 0;

		while(true) {
			assert(i < table_size);

			auto idx = hash_to_index(k + i);
			auto &c = cells[idx];

			if(!c.valid)
				return std::nullopt;

			if(c.key == k)
				return c.value;

			++i;
		}
	}

    // We must rehash all our entries in order to resize the hashtable
    void resize_table() {
        // Save old state
        cell * old_cells = cells;
        auto old_size = table_size;

        // Double the size
        table_size = table_size << 1u;
        cells = new cell[table_size];

        // Rehash existing entries in new table
        for(size_t i = 0; i < old_size; ++i) {
			auto &c = old_cells[i];
            if(c.valid) {
                this->put(c.key, c.value);
            }
        }
        delete [] old_cells;
    }
};

// This hash table uses linear probing, uses the scaling strategy to determine a cell and resizes dynamicly
struct dynamic_scaling_table {
	static constexpr const char *name = "dynamic_scaling";

	struct cell {
		int key;
		int value;
		bool valid = false;
	};

    size_t table_size = 8;
    // table_size = 2^exp
    unsigned int exp = 3;
    size_t insertions = 0;
	cell *cells = nullptr;
    float max_fill;

    // Initidalize our array with 8 cells
    dynamic_scaling_table(float max_fill) : cells{new cell[table_size]{}} {
        this->max_fill = max_fill;
    }

	// Note: to simply the implementation, destructors and copy/move constructors are missing.

    // This function computes floor(h * m / 2 ^ 31). Since m = 2 ^ exp, we can do everything as bitshift
	// As hash, we just use the integer key directly.
    int scaling(int h) {
        // if the table is bigger than 2^31, this implementation won't work because
        // of the following bitshift and because the index 'idx' will overflow
        assert(31u - exp > 0);
        return h >> (31u - exp);
    }

	int hash_to_index(int h) {
		return h & (table_size - 1);
	}

	void put(int k, int v) {
		unsigned int i = 0;

		while(true) {
			assert(i < table_size);
            
			auto idx = hash_to_index(scaling(k) + i);
			auto &c = cells[idx];

			if(!c.valid) {
				c.key = k;
				c.value = v;
				c.valid = true;
                ++insertions;
                if((float)insertions / table_size > max_fill) {
                    resize_table();
                }
				return;
			}

			if(c.key == k) {
				c.value = v;
				return;
			}

			++i;
		}
	}

	std::optional<int> get(int k) {
		int i = 0;

		while(true) {
			assert(i < table_size);

			auto idx = scaling(k) + i;
			auto &c = cells[idx];

			if(!c.valid)
				return std::nullopt;

			if(c.key == k)
				return c.value;

			++i;
		}
	}

    // We must rehash all our entries in order to resize the hashtable
    void resize_table() {
        // Save old state
        cell * old_cells = cells;
        auto old_size = table_size;

        // Double the size
        table_size = table_size << 1u;
        ++exp;
        cells = new cell[table_size];

        // Rehash existing entries in new table
        for(size_t i = 0; i < old_size; ++i) {
			auto &c = old_cells[i];
            if(c.valid) {
                this->put(c.key, c.value);
            }
        }
        //delete [] old_cells;
    }
};

// For comparsion, an implementation that uses std::unordered_map.
// This is apples-to-oranges since std::unordered_map does not respect our fill factor.
struct stl_table {
	static constexpr const char *name = "stl";

	stl_table(float max_fill) {
		(void)max_fill; // Ignore the fill factor.
	}

	void put(int k, int v) {
		map[k] = v;
	}

	std::optional<int> get(int k) {
		auto it = map.find(k);
		if(it == map.end())
			return std::nullopt;
		return it->second;
	}

	std::unordered_map<int, int> map;
};

// Helper function to perform a microbenchmark.
// You should not need to touch this.
template<typename Algo>
void microbenchmark(float max_fill) {
	Algo table{max_fill};

	std::mt19937 prng{42};
	std::uniform_int_distribution<int> distrib;

	std::cerr << "Running microbenchmark..." << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	int nv = 1;
	for(int i = 0; i < num_insertions; ++i)
		table.put(distrib(prng), nv++);
	auto t = std::chrono::high_resolution_clock::now() - start;

	std::cout << "algo: " << Algo::name << std::endl;
	std::cout << "max_fill: " << max_fill << std::endl;
	std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t).count()
			<< " # ms" << std::endl;
}

static const char *usage_text =
	"Usage: hashing [OPTIONS]\n"
	"Possible OPTIONS are:\n"
	"    --algo ALGORITHM\n"
	"        Select an algorithm {static,dynamic_modulo,dynamic_scaling,stl}.\n"
	"    --max-fill FACTOR\n"
	"        Set the maximal fill factor before the table grows.\n";

int main(int argc, char **argv) {
	std::string_view algorithm;
	float max_fill = 0.5;

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
		}else if(handle_unary_option("--max-fill")) {
			max_fill = std::atof(arg);
		}else{
			error("unknown command line option");
		}
	}

	if(*p)
		error("unexpected arguments");

	// Verify that options are correct and run the algorithm.

	if(algorithm.empty())
		error("no algorithm specified");

	if(algorithm == "static") {
		microbenchmark<static_table>(max_fill);
	}else if(algorithm == "dynamic_modulo") {
		microbenchmark<dynamic_modulo_table>(max_fill);
	}else if(algorithm == "dynamic_scaling") {
		microbenchmark<dynamic_scaling_table>(max_fill);
	}else if(algorithm == "stl") {
		microbenchmark<stl_table>(max_fill);
	}else{
		error("unknown algorithm");
	}
}
