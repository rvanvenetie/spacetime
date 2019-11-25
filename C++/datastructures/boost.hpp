#pragma once
#include <boost/container/deque.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/container/stable_vector.hpp>
#include <boost/container/static_vector.hpp>
template <typename I, size_t N>
using SmallVector = boost::container::small_vector<I, N>;

template <typename I, size_t N>
using StaticVector = boost::container::static_vector<I, N>;

typedef boost::container::deque_options<boost::container::block_size<128>>::type
    deque_block_128_option_t;
