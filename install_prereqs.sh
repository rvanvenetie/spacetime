#!/bin/bash
sudo apt-get purge cmake cmake-data libboost-dev  -y
sudo apt-get autoremove -y

sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo add-apt-repository ppa:mhier/libboost-latest -y
https://launchpad.net/~nschloe/+archive/ubuntu/eigen-backports -y
sudo apt-get update -y

sudo apt-get install gcc-7 g++-7 libboost-1.70-dev eigen3 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc

pip3 install cmake
