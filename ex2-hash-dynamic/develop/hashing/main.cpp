#include <algorithm>
#include <cassert>
#include <cstring>
#include <chrono>
#include <iostream>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>
#include <tuple>
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

	static_table(float max_fill, unsigned subtables)
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

    // Initialize our array with 8 cells
    dynamic_modulo_table(float max_fill, unsigned subtables) : cells{new cell[table_size]{}} {
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
    unsigned exp = 3;
    size_t insertions = 0;
	cell *cells = nullptr;
    float max_fill;

    // Initialize our array with 8 cells
    dynamic_scaling_table(float max_fill, unsigned subtables) : cells{new cell[table_size]{}} {
        this->max_fill = max_fill;
    }

	// Note: to simply the implementation, destructors and copy/move constructors are missing.

    // This function computes floor(h * m / 2 ^ 31). Since m = 2 ^ exp, we can do everything as bitshift
	// As hash, we just use the integer key directly.
    int scaling(int h) {
        // if the table is bigger than 2^31, this implementation won't work because
        // of the following bitshift and because the index 'idx' will overflow
        assert(exp < 32);
        return h >> (31u - exp);
    }

	int hash_to_index(int h) {
		return h & (table_size - 1);
	}

	void put(int k, int v) {
		unsigned i = 0;
        auto scaled_k = scaling(k);

		while(true) {
			assert(i < table_size);
            
			auto idx = hash_to_index(scaled_k + i);
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
		unsigned i = 0;
        auto scaled_k = scaling(k);

		while(true) {
			assert(i < table_size);

			auto idx = hash_to_index(scaled_k + i);
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
        delete [] old_cells;
    }
};

// This hash table uses linear probing and subtables
struct dysect_table {
	static constexpr const char *name = "dysect";

	struct cell {
		int key;
		int value;
		bool valid = false;
	};

    const unsigned subtables;
    const float max_fill;
	cell **cells = nullptr;

    // This tuple array stores the size and number of inserted elements for each subtable
    std::tuple<size_t, size_t> * fill_stat;

    dysect_table(float max_fill, unsigned subtables) : subtables{subtables}, max_fill{max_fill} {
        // Create subtables and status array
        cells = new cell * [subtables]();
        fill_stat = new std::tuple<size_t, size_t>[subtables];
        
        // Initialize subtables and status array
        for(int i = 0; i < subtables; i++) {
            // Start with 8 cells for each subtable
            cells[i] = new cell[8];

            // Each subtable has 8 cells and 0 inserted elements
            fill_stat[i] = std::make_tuple(8,0);
        }
    }

    // Compute how many bits are needed to represent num = 2^x
    // Call with function only with powers of 2!!
    unsigned bits(unsigned num) {
        int bits = 0;
        for (; num != 1; num >>= 1, bits++);
        return bits;
    }

    int hash_to_index(unsigned h, size_t table_size) {
        return h & (table_size - 1u);
    }

    float fill_factor(unsigned subtable_index) {
        return (float)std::get<1>(fill_stat[subtable_index]) / std::get<0>(fill_stat[subtable_index]);
    }

	void put(int k, int v) {
        unsigned i = 0;
        // extract first t bits of our key to get our subtable index
        unsigned subtable_index = k >> (31 - bits(subtables));
        auto subtable_size = std::get<0>(fill_stat[subtable_index]);

        while(true) {
            // assert i < size of subtable
            assert(i < subtable_size);
            
            auto idx = hash_to_index(k + i, subtable_size);
            auto subtable = (cell *) cells[subtable_index];
            auto &c = subtable[idx];

            if(!c.valid) {
                c.key = k;
                c.value = v;
                c.valid = true;

                // Increment insertion counter
                std::get<1>(fill_stat[subtable_index]) += 1;

                if(fill_factor(subtable_index) > max_fill) {
                    // reset insertion counter
                    std::get<1>(fill_stat[subtable_index]) = 0;
                    resize_table(subtable_index);
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
        unsigned i = 0;
        unsigned subtable_index = k >> (31 - bits(subtables));
        auto subtable_size = std::get<0>(fill_stat[subtable_index]);

        while(true) {
            assert(i < subtable_size);

            auto idx = hash_to_index(k + i, subtable_size);
            auto subtable = (cell *) cells[subtable_index];
            auto &c = subtable[idx];

            if(!c.valid)
                return std::nullopt;

            if(c.key == k)
                return c.value;

            ++i;
        }
    }

    // // We must rehash all our entries in order to resize the hashtable
    void resize_table(unsigned subtable_index) {

        // Save old state
        auto old_subtable = (cell *) cells[subtable_index];
        auto old_size = std::get<0>(fill_stat[subtable_index]);

        // Double the size
        cells[subtable_index] = new cell[old_size * 2];
        std::get<0>(fill_stat[subtable_index]) *= 2;

        // Rehash existing entries in new table
        for(size_t i = 0; i < old_size; ++i) {
            auto &c = old_subtable[i];
            if(c.valid) {
                this->put(c.key, c.value);
            }
        }
        delete [] old_subtable;
    }
};


// For comparsion, an implementation that uses std::unordered_map.
// This is apples-to-oranges since std::unordered_map does not respect our fill factor.
struct stl_table {
	static constexpr const char *name = "stl";

	stl_table(float max_fill, unsigned subtables) {
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
void microbenchmark(float max_fill, unsigned subtables) {
	Algo table{max_fill, subtables};

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
	std::cout << "subtables: " << subtables << std::endl;
	std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t).count()
			<< " # ms" << std::endl;
}

static const char *usage_text =
	"Usage: hashing [OPTIONS]\n"
	"Possible OPTIONS are:\n"
	"    --algo ALGORITHM\n"
	"        Select an algorithm {static,dynamic_modulo,dynamic_scaling,dysect,stl}.\n"
	"    --max-fill FACTOR\n"
	"        Set the maximal fill factor before the table grows.\n"
	"    --subtables NUMBER\n"
	"        Set the number of subtables that is used by DySECT\n";

int main(int argc, char **argv) {
	std::string_view algorithm;
	float max_fill = 0.5;
    unsigned subtables = 1;

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
		}else if(handle_unary_option("--subtables")) {
			subtables = std::atoi(arg);
		}else{
			error("unknown command line option");
		}
	}

	if(*p)
		error("unexpected arguments");

	// Verify that options are correct and run the algorithm.

	if(algorithm.empty())
		error("no algorithm specified");
    if(subtables != 1 && subtables != 2 && subtables != 4 
            && subtables != 8 && subtables != 16 && subtables != 32)
        error("The following values are allowed for subtables: [1 2 4 8 16 32]");

	if(algorithm == "static") {
		microbenchmark<static_table>(max_fill, subtables);
	}else if(algorithm == "dynamic_modulo") {
		microbenchmark<dynamic_modulo_table>(max_fill, subtables);
	}else if(algorithm == "dynamic_scaling") {
		microbenchmark<dynamic_scaling_table>(max_fill, subtables);
	}else if(algorithm == "dysect") {
		microbenchmark<dysect_table>(max_fill, subtables);
	}else if(algorithm == "stl") {
		microbenchmark<stl_table>(max_fill, subtables);
	}else{
		error("unknown algorithm");
	}
}
