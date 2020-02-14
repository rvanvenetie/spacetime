#!/bin/bash

# Remove existing cmake.
sudo apt-get purge cmake cmake-data  -y

# Add repository for the latest gcc.
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y

# Install latest gcc.
install-package --update gcc-9 g++-9 snapd
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 130 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Install latest cmake.
sudo snap install cmake --classic

# Build.
cmake -S C++ -B C++/build
