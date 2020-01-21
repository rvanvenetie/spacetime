#!/bin/bash
sudo apt-get purge cmake cmake-data libboost-dev  -y
sudo apt-get autoremove -y

sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo add-apt-repository ppa:mhier/libboost-latest -y
sudo add-apt-repository ppa:nschloe/eigen-backports -y

install-package gcc-9 g++-9 libboost1.70-dev libeigen3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 130 --slave /usr/bin/g++ g++ /usr/bin/g++-9

pip3 install cmake

mkdir C++/build
cd C++/build
cmake ..
make
cd ../..
