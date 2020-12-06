project('expalgo-hashing', 'cpp')

omp = dependency('openmp')

executable('hashing',
	'main.cpp',
	override_options: ['cpp_std=c++17'],
	dependencies: omp,
	install: true)
