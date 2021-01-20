project('expalgo-csr', 'cpp', 'cuda')

omp = dependency('openmp')

executable('transpose',
	'transpose.cpp',
	dependencies: omp,
	override_options: ['cpp_std=c++17'],
	install: true)

executable('sssp-cpu',
	'sssp-cpu.cpp',
	dependencies: omp,
	override_options: ['cpp_std=c++17'],
	install: true)

executable('sssp-gpu',
	'sssp-gpu.cu',
	override_options: ['cpp_std=c++17'],
	install: true)
