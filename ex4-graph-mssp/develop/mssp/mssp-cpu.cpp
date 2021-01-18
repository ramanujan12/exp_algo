#include <omp.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <random>

#include "csr.hpp"

struct dijkstra {
	void run(const csr_matrix &mat, unsigned int s) {
		d.resize(mat.n);

		std::fill(d.begin(), d.end(), FLT_MAX);
		d[s] = 0;
		pq.push(s);

		while(!pq.empty()) {
			auto u = pq.extract_top();

			// Iterate over the neighbors in CSR style.
			for(unsigned int i = mat.ind[u]; i < mat.ind[u + 1]; ++i) {
				auto v = mat.cols[i];
				auto weight = mat.weights[i];
				
				if(d[v] > d[u] + weight) {
					d[v] = d[u] + weight;
					pq.update(v);
				}
			}
		}
	}

private:
	class compare {
	public:
		bool operator() (unsigned int u, unsigned int v) const {
			return self->d[u] < self->d[v];
		}

		dijkstra *self;
	};

	std::vector<float> d;
	DAryAddressableIntHeap<unsigned int, 2, compare> pq{compare{this}};
};

struct batch_bellman_ford {
	batch_bellman_ford(unsigned int batchsize) {
		// TODO
	}

	void run(const csr_matrix &tr, const std::vector<unsigned int> &sources) {
		// TODO
	}

private:
	std::vector<float> d;
	std::vector<float> d_new;
};

int main(int argc, char **argv) {
	if(argc != 4)
		throw std::runtime_error("Expected algorithm, instance and batch size as argument");

	unsigned int batchsize = std::atoi(argv[3]);

	std::mt19937 prng{42};
	std::uniform_real_distribution<float> weight_distrib{0.0f, 1.0f};

	// Load the graph.
	std::cout << "instance: " << argv[2] << std::endl;
	std::cout << "threads: " << omp_get_max_threads() << std::endl;
	std::cout << "batchsize: " << batchsize << std::endl;

	std::ifstream ins(argv[2]);
	std::vector<std::tuple<unsigned int, unsigned int, float>> cv;

	auto io_start = std::chrono::high_resolution_clock::now();
	read_graph_unweighted(ins, [&] (unsigned int u, unsigned int v) {
		// Generate a random edge weight in [a, b).
		cv.push_back({u, v, weight_distrib(prng)});
	});

	auto mat = coordinates_to_csr(std::move(cv));
	auto t_io = std::chrono::high_resolution_clock::now() - io_start;

	std::cout << "time_io: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t_io).count() << std::endl;
	std::cout << "n_nodes: " << mat.n << std::endl;
	std::cout << "n_edges: " << mat.nnz << std::endl;

	// Compute the transpose (for Bellman-Ford).
	auto tr = transpose(mat);

	// Generate random sources.
	std::uniform_int_distribution<unsigned int> s_distrib{0, mat.n - 1};
	std::vector<unsigned int> sources;
	for(unsigned int i = 0; i < batchsize; ++i)
		sources.push_back(s_distrib(prng));

	// Run the algorithm.
	auto algo_start = std::chrono::high_resolution_clock::now();
	if(!strcmp(argv[1], "parallel-dijkstra")) {
		#pragma omp parallel
		{
			dijkstra algo;

			#pragma omp for
			for(unsigned int i = 0; i < batchsize; ++i)
				algo.run(mat, sources[i]);
		}
	}else if(!strcmp(argv[1], "batch-bf")) {
		batch_bellman_ford algo{batchsize};
		algo.run(tr, sources);
	}else{
		throw std::runtime_error("Unexpected algorithm");
	}
	auto t_algo = std::chrono::high_resolution_clock::now() - algo_start;

	std::cout << "time_mssp: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t_algo).count() << std::endl;
}
