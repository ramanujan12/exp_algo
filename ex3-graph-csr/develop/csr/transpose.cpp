#include <omp.h>

#include <chrono>
#include <fstream>
#include <random>

#include "csr.hpp"

int main(int argc, char **argv) {
	if(argc != 2)
		throw std::runtime_error("Expected instance as argument");

	std::mt19937 prng{42};
	std::uniform_real_distribution<float> distrib{0.0f, 1.0f};

	// Load the graph.
	std::cout << "instance: " << argv[1] << std::endl;
	std::cout << "threads: " << omp_get_max_threads() << std::endl;

	std::ifstream ins(argv[1]);
	std::vector<std::tuple<unsigned int, unsigned int, float>> cv;

	auto io_start = std::chrono::high_resolution_clock::now();
	read_graph_unweighted(ins, [&] (unsigned int u, unsigned int v) {
		// Generate a random edge weight in [a, b).
		cv.push_back({u, v, distrib(prng)});
	});

	auto mat = coordinates_to_csr(std::move(cv));
	auto t_io = std::chrono::high_resolution_clock::now() - io_start;

	std::cout << "time_io: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t_io).count() << std::endl;
	std::cout << "n_nodes: " << mat.n << std::endl;
	std::cout << "n_edges: " << mat.nnz << std::endl;

	auto algo_start = std::chrono::high_resolution_clock::now();
	auto tr = transpose(mat);
	auto t_algo = std::chrono::high_resolution_clock::now() - algo_start;

	std::cout << "time_transpose: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t_algo).count() << std::endl;
}
