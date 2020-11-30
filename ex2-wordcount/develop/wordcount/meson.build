project('expalgo-wordcount', 'cpp')

omp = dependency('openmp')

executable('wordcount',
	'main.cpp',
	override_options: ['cpp_std=c++17'],
	dependencies: omp,
	install: true)
