/*
  to-do : 1. change the matrix order to column major and
             -> change matrix 
	     -> change indexing
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include "csr.hpp"

//__________________________________________________________________
// device function for the indexing
// rather a lambda?
__device__ unsigned int crd2idx(unsigned int batch,
				unsigned int batchsize,
				unsigned int v) {
  return batch * batchsize + v;
}

//__________________________________________________________________
// produce a 2d grid based on the sizes
dim3 get_grid(unsigned int x, unsigned int y, dim3 block_2d) {
  dim3 grid_2d((x + (block_2d.x - 1)) / block_2d.x,
	       (y + (block_2d.y - 1)) / block_2d.y);
  return grid_2d;
}

//__________________________________________________________________
// produce a grid_block based on the sizes
/*
dim3 get_block(unsigned int x, unsigned int y) {  
}
*/
//__________________________________________________________________
// gpu function to run the multiple source bellman ford
// 1. use array of things indices to be run
// 2. loop over indices to be run
// 3. give back array of the ones that changed
/*
__global__ void bf_iteration(int           n,
			     unsigned int  batchsize,
			     unsigned int *csr_index,
			     unsigned int *csr_cols,
			     float        *csr_weights,
			     float        *d,
			     float        *d_new,
			     unsigned int *ind,
			     int          *result) {
  auto thisThread = blockIdx.x * blockDim.x + threadIdx.x;
  auto numThreads = gridDim.x + blockDim.x;

  // loop over all the batches that need to be done
  for (unsigned int batch = 0; batch < batchsize; ++batch) {
    bool changes = false;
    auto idx = ind[batch];
    for (unsigned int v = thisThread; v < n; v += numThreads) {
      float dist = d[crd2idx(idx, batchsize, v)];
      for(unsigned int i = csr_index[v]; i < csr_index[v + 1]; ++i) {
	auto u = csr_cols[i];
	auto weight = csr_weights[i];
	
	if(dist > d[crd2idx(idx, batchsize, u)] + weight) {
	  dist = d[crd2idx(idx, batchsize, u)] + weight;
	  changes = true;
	}
      }
      d_new[crd2idx(idx, batchsize, v)] = dist;
    }
    // check if a certain batch changed
    if (changes) {
      result[idx] = 1;
    }
  }
}
*/

__global__ void bf_iteration_2d(int           n,
				unsigned int  batchsize,
				unsigned int *csr_index,
				unsigned int *csr_cols,
				float        *csr_weights,
				float        *d,
				float        *d_new,
				int          *result) {
  auto thread_x = blockIdx.x * blockDim.x + threadIdx.x;
  auto thread_y = blockIdx.y * blockDim.y + threadIdx.y;
  auto n_threads_x = gridDim.x + blockDim.x;
  auto n_threads_y = gridDim.y + blockDim.y;
  
  // loop over all the batches that need to be done
  bool changes = false;
  for (unsigned int batch = thread_y; batch < batchsize; batch += n_threads_y) {
    for (unsigned int v = thread_x; v < n; v += n_threads_x) {
      float dist = d[crd2idx(batch, batchsize, v)];
      for(unsigned int i = csr_index[v]; i < csr_index[v + 1]; ++i) {
	auto u = csr_cols[i];
	auto weight = csr_weights[i];
	
	if(dist > d[crd2idx(batch, batchsize, u)] + weight) {
	  dist = d[crd2idx(batch, batchsize, u)] + weight;
	  changes = true;
	}
      }
      d_new[crd2idx(batch, batchsize, v)] = dist;
    }
    // check if a certain batch changed
  }
  if (changes) {
    *result = 1;
  }
}

//___________________________________________________________________
// run the bf stuff
void run_bf(const csr_matrix                &tr,
	    unsigned int                     batchsize,
	    const std::vector<unsigned int> &sources) {
  // 1.0. allocate memory matrix and move to gpu
  unsigned int *csr_index;
  unsigned int *csr_cols;
  float        *csr_weights;

  cudaMalloc(&csr_index, (tr.n + 1) * sizeof(unsigned int));
  cudaMalloc(&csr_cols,      tr.nnz * sizeof(unsigned int));
  cudaMalloc(&csr_weights,   tr.nnz * sizeof(float));
  
  cudaMemcpy(csr_index,   tr.ind.data(), (tr.n + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(csr_cols,    tr.cols.data(),    tr.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(csr_weights, tr.weights.data(), tr.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);

  // 1.1 allocate memory distances and move to gpu
  float *d;
  float *d_new;
  int   *result;
  cudaMalloc(&d,      batchsize * tr.n * sizeof(float));
  cudaMalloc(&d_new,  batchsize * tr.n * sizeof(float));
  cudaMalloc(&result, /*batchsize * */   sizeof(int));

  std::vector <float> initial;
  initial.resize(tr.n * batchsize);  
  std::fill(initial.begin(), initial.end(), FLT_MAX);
  for (std::size_t b = 0; b < batchsize; ++b) {
    initial[b*batchsize + sources[b]] = 0;
  }
  
  cudaMemcpy(d, initial.data(), tr.n * batchsize * sizeof(float), cudaMemcpyHostToDevice);

  /*
  // 2. loop over all the problems until they are all solved
  // controll array c for the indices that did change
  // array of indices to run over
  unsigned int *c, *ind_host, *ind_dev;
  c = (unsigned int*) malloc (batchsize * sizeof(unsigned int));
  ind_host = (unsigned int*) malloc (batchsize * sizeof(unsigned int));
  for (unsigned int i = 0; i < batchsize; ++i) {
    ind_host[i] = i;
  }
  cudaMalloc(&ind_dev, batchsize*sizeof(unsigned int));
  
  unsigned int num_blocks = (tr.n + 255) / 256;
  unsigned int to_solve = batchsize;
  while(true) {
    cudaMemset(result,  0, batchsize*sizeof(int));
    cudaMemcpy(ind_dev, ind_host, batchsize*sizeof(int), cudaMemcpyHostToDevice);
    bf_iteration<<<num_blocks, 256>>>(tr.n, to_solve,
				      csr_index, csr_cols, csr_weights,
				      d, d_new, ind_dev, result);
    
    // check for iteration and decide which ones should be iterated again
    cudaMemcpy(c, result, batchsize*sizeof(int), cudaMemcpyDeviceToHost);
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < batchsize; ++i) {
      if (!c[i]) {
	ind_host[cnt] = i;
	++cnt;
      }
    }
    to_solve = cnt;
    if (cnt == batchsize)
      break;
    std::swap(d, d_new);
  }
  */
  // 2d strategy
  dim3 block_2d(1024, 1024);
  dim3 grid_2d = get_grid(tr.n, batchsize, block_2d);
  
  while(true) {
    cudaMemset(result, 0, sizeof(int));
    bf_iteration_2d<<<grid_2d, block_2d>>>(tr.n, batchsize,
					   csr_index, csr_cols, csr_weights,
					   d, d_new, result);
    unsigned int c;
    cudaMemcpy(&c, result, sizeof(int), cudaMemcpyDeviceToHost);
    if (!c)
      break;
    std::swap(d, d_new);
  }
  
  // 4. free memory
  cudaFree(csr_index);
  cudaFree(csr_cols);
  cudaFree(csr_weights);
  cudaFree(d);
  cudaFree(d_new);
  cudaFree(result);
  // cudaFree(ind_dev);
  
  // free(c);
  // free(ind_host);
}

//___________________________________________________________________
// int main(int argc, char** argv)
int main(int argc, char **argv) {
  if(argc != 3)
    throw std::runtime_error("Expected instance and batch size as argument");
  
  unsigned int batchsize = std::atoi(argv[2]);
  
  std::mt19937 prng{42};
  std::uniform_real_distribution<float> weight_distrib{0.0f, 1.0f};
  
  // Load the graph.
  std::cout << "algo: " << "bf_gpu" << std::endl;
  std::string instance(argv[1]);
  std::size_t npos = instance.find_last_of("/");
  instance = instance.substr(npos+1);
  std::cout << "instance: " << instance << std::endl;
  std::cout << "batchsize: " << batchsize << std::endl;
  
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
  
  auto tr = transpose(std::move(mat));
  
  // Generate random sources.
  std::uniform_int_distribution<unsigned int> s_distrib{0, mat.n - 1};
  std::vector<unsigned int> sources;
  for(unsigned int i = 0; i < batchsize; ++i)
    sources.push_back(s_distrib(prng));
  
  // Run the algorithm.
  auto algo_start = std::chrono::high_resolution_clock::now();
  run_bf(tr, batchsize, sources);
  auto t_algo = std::chrono::high_resolution_clock::now() - algo_start;
  
  std::cout << "time_mssp: "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(t_algo).count() << std::endl;
}
