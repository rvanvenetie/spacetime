
#include "sparse_vector.hpp"

#include <cmath>
#include <set>
#include <unordered_map>

#include "bases.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace Time {

TEST(SparseVector, BLAS) {
  // Reset the persistent trees.
  Bases B;

  int ml = 7;
  B.three_point_tree.UniformRefine(ml);
  auto nodes = B.cont_lin_tree.meta_root->Bfs();

  // First sum up two disjoint subsets.
  SparseVector<ContLinearScalingFn> vec1, vec2, vecsum;
  for (auto node : nodes) {
    if (node->index() % 2)
      vec1.emplace_back(node, 1);
    else
      vec2.emplace_back(node, -1);
  }
  vecsum += vec1;
  vecsum += vec2;
  for (auto [phi, coeff] : vecsum) {
    if (phi->index() % 2)
      ASSERT_EQ(coeff, 1);
    else
      ASSERT_EQ(coeff, -1);
  }

  // Now sum up itself, this has overlap!
  vecsum += vecsum;
  for (auto [phi, coeff] : vecsum) {
    if (phi->index() % 2)
      ASSERT_EQ(coeff, 2);
    else
      ASSERT_EQ(coeff, -2);
  }

  // Now sum up everything by appending it.
  SparseVector<ContLinearScalingFn> vecsum_major;
  vecsum_major.insert(vecsum_major.end(), vec1.begin(), vec1.end());
  vecsum_major.insert(vecsum_major.end(), vec2.begin(), vec2.end());
  vecsum_major.insert(vecsum_major.end(), vecsum.begin(), vecsum.end());
  vecsum_major.Compress();
  for (auto [phi, coeff] : vecsum_major) {
    if (phi->index() % 2)
      ASSERT_EQ(coeff, 3);
    else
      ASSERT_EQ(coeff, -3);
  }

  // Lastly, try to multiply.
  vecsum_major *= 1.0 / 3;
  for (auto [phi, coeff] : vecsum_major) {
    if (phi->index() % 2)
      ASSERT_EQ(coeff, 1);
    else
      ASSERT_EQ(coeff, -1);
  }
}
}  // namespace Time
