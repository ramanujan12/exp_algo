#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include "csr.hpp"

__global__ void bf_iteration(int n,
		unsigned int *csr_index, unsigned int *csr_cols, float *csr_weights,
		float *d, float *d_new, int *result) {
	auto thisThread = blockIdx.x * blockDim.x + threadIdx.x;
	auto numThreads = gridDim.x + blockDim.x;

	bool changes = false;
	for (unsigned int v = thisThread; v < n; v += numThreads) {
		float dist = d[v];
		for(unsigned int i = csr_index[v]; i < csr_index[v + 1]; ++i) {
			auto u = csr_cols[i];
			auto weight = csr_weights[i];

			if(dist > d[u] + weight) {
				dist = d[u] + weight;
				changes = true;
			}
		}
		d_new[v] = dist;
	}
	if(changes)
		*result = 1;
}

/*int main(void) {
	int n = 2;
	// Matrix, 1st row: (1, 1).
	csr_index[0] = 0;
	csr_cols[0] = 0; csr_weights[0] = 1;
	csr_cols[1] = 1; csr_weights[1] = 1;

	// Matrix, 2nd row: (0, 1).
	csr_index[1] = 2;
	csr_cols[2] = 1; csr_weights[2] = 1;

	// Matrix, past-the-end.
	csr_index[2] = 3;

	// Input vector.
	v[0] = 42;
	v[1] = 21;

	// Run kernel: 1 thread block, 2 threads/block.
	bf_iteration<<<1, 2>>>(n, csr_index, csr_cols, csr_weights, v, out);

	// Wait for the kernel to finish.
	cudaDeviceSynchronize();

	std::cout << "result: (" << out[0] << ", " << out[1] << ")" << std::endl;

	cudaFree(csr_index);
	cudaFree(csr_cols);
	cudaFree(csr_weights);
	cudaFree(v);
	cudaFree(out);

	return 0;
}
*/

int main(int argc, char **argv) {
	if(argc != 3)
		throw std::runtime_error("Expected instance and number of sources as argument");

	unsigned int n_sources = std::atoi(argv[2]);

	std::mt19937 prng{42};
	std::uniform_real_distribution<float> weight_distrib{0.0f, 1.0f};

	// Load the graph.
	std::cout << "instance: " << argv[1] << std::endl;
	std::cout << "n_sources: " << n_sources << std::endl;

	std::ifstream ins(argv[1]);
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

	std::uniform_int_distribution<unsigned int> s_distrib{0, mat.n - 1};

	auto tr = transpose(std::move(mat));

	unsigned int num_blocks = (tr.n + 255) / 256;

	unsigned int *csr_index;
	unsigned int *csr_cols;
	float *csr_weights;
	float *d;
	float *d_new;
	int *result;

	cudaMalloc(&csr_index, (tr.n + 1) * sizeof(unsigned int));
	cudaMalloc(&csr_cols, tr.nnz * sizeof(unsigned int));
	cudaMalloc(&csr_weights, tr.nnz * sizeof(float));
	cudaMalloc(&d, tr.n * sizeof(float));
	cudaMalloc(&d_new, tr.n * sizeof(float));
	cudaMalloc(&result, sizeof(int));

	cudaMemcpy(csr_index, tr.ind.data(), (tr.n + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_cols, tr.cols.data(), tr.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_weights, tr.weights.data(), tr.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);

	auto algo_start = std::chrono::high_resolution_clock::now();
	std::vector<float> initial;
	initial.resize(tr.n);
	for(unsigned int i = 0; i < n_sources; ++i) {
		auto s = s_distrib(prng);

		std::fill(initial.begin(), initial.end(), FLT_MAX);
		initial[s] = 0;
		cudaMemcpy(d, initial.data(), tr.n * sizeof(float), cudaMemcpyHostToDevice);

		while(true) {
			cudaMemset(result, 0, sizeof(int));
			bf_iteration<<<num_blocks, 256>>>(tr.n, csr_index, csr_cols, csr_weights,
					d, d_new, result);

			unsigned int c;
			cudaMemcpy(&c, result, sizeof(int), cudaMemcpyDeviceToHost);
			if(!c)
				break;
			std::swap(d, d_new);
		}
	}
	auto t_algo = std::chrono::high_resolution_clock::now() - algo_start;

	std::cout << "time_sssp: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t_algo).count() << std::endl;

	cudaFree(csr_index);
	cudaFree(csr_cols);
	cudaFree(csr_weights);
	cudaFree(d);
	cudaFree(d_new);
	cudaFree(result);
}
