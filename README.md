# Efficient space-time adaptivity for parabolic evolution equations using wavelets in time and finite elements in space
This repository contains the implementation that was used to generate the numerics in arXiv:2104.08143 and arXiv:2101.03956.

This project provides a linear complexity implementation of a space-time
adaptive solver for parabolic evolution equations.  Aiming at a truly linear-complexity 
implementation, we implemented this algorithm using tree-based algorithms, without the use of hash maps.

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
