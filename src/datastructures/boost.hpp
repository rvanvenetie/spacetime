#pragma once
#include <boost/container/deque.hpp>
#include <boost/container/options.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <vector>
template <typename I, size_t N>
using SmallVector = boost::container::small_vector<I, N>;

template <typename I, size_t N>
using StaticVector = boost::container::static_vector<I, N>;

// Boost deque container with default block size N. (REQUIRES LATEST BOOST).
template <typename I, size_t N = 128>
using Deque =
    boost::container::deque<I, void,
                            typename boost::container::deque_options<
                                boost::container::block_size<N>>::type>;
