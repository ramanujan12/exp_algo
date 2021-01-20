project('expalgo-hashing', 'cpp')

omp = dependency('openmp')

executable('dijkstra',
	'd_ary_addressable_int_heap.hpp',
	'main.cpp',
	override_options: ['cpp_std=c++17'],
	link_args: '-lstdc++fs',
	dependencies: omp,
	install: true)
