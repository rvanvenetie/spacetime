# Efficient space-time adaptivity for parabolic evolution equations using wavelets in time and finite elements in space
This repository contains an implementation of arXiv:2101.03956, as described in.

This project provides a linear complexity implementation of a space-time
adaptive solver for parabolic evolution equations.  The trial spaces that
we consider here are given as sparse tensor product approximations
of wavelets-in-time and (locally refined) finite element spaces-in-space.
Special care has to be taken to evaluate matrix-vector products, as the
system-matrix w.r.t. such a multi-level type basis is not sparse. By restricting
to bases that are spanned by tensor products having index sets that form double-trees,
we can still evaluate the system matrices in linear complexity.

Aiming at a truly linear-complexity implementation, we implemented this algorithm
using tree-based algorithms, without the use of hash maps.

## Requirements
- A C++17 compliant compiler
- CMake, version >= 3.15

## Install instructions
Download the release or clone this repository. Then, build using CMake:

```bash
cd spacetime
cmake -S src -B build
cmake --build build
```

The tests can be run using `make -C build test`. The adapative method,
used for the numerical results in the paper, is given in
`src/applications/adaptive.cpp`. It can be run using the flags specified in the source, e.g.
```bash
build/applications/adaptive --problem singular --mark_theta 0.5 --domain l-shape
```


## License
This project is licensed under the terms of the [MIT license](LICENSE.md).
