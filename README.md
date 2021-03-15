# A wavelet-in-time, finite element-in-space adaptive method for parabolic evolution equations
This repository contains an implementation of arXiv:2101.03956, as described in.

This project provides a linear complexity implementation of a space-time
adaptive solver for parabolic evolution equations.  The trial spaces that
we consider here are given as sparse tensor product approximations
of wavelets-in-time and (locally refined) finite element spaces-in-space.
Special care has to be taken to evaluate matrix-vector products, as the
system-matrix w.r.t. such a multi-level type basis is not sparse. By restricting
to bases that are span by tensor products having index sets that form double-trees,
we can still evaluatie system matrices in linear complexity.

Aiming at a truly linear-complexity implementation, we implemented this algorithm
using tree-based algorithms, without the use of hash maps.

## Requirements
- A C++17 compliant compiler
- CMake, version >= 3.15

## Install instructions
Download the release or clone this repository.  Then, build using CMake :

```bash
cd spacetime
cmake -S src -B build
cmake --build build
```

## License
This project is licensed under the terms of the [MIT license](LICENSE.md).
