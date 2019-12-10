#include <cmath>
#include <map>
#include <set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "haar_basis.hpp"

namespace Time {
using ::testing::ElementsAre;

TEST(HaarBasis, functions) {
  ASSERT_EQ(disc_cons_tree.meta_root->children().size(), 1);
  ASSERT_EQ(haar_tree.meta_root->children().size(), 1);

  auto mother_scaling = disc_cons_tree.meta_root->children()[0];
  ASSERT_EQ(mother_scaling->labda(), std::make_pair(0, 0));
  ASSERT_EQ(mother_scaling->Eval(0.25), 1.0);
  ASSERT_EQ(mother_scaling->Eval(0.85), 1.0);

  auto mother_wavelet = haar_tree.meta_root->children()[0];
  mother_wavelet->Refine();
  ASSERT_EQ(mother_wavelet->children().size(), 1);
  mother_wavelet = mother_wavelet->children()[0];
  ASSERT_EQ(mother_wavelet->Eval(0.25), 1.0);
  ASSERT_EQ(mother_wavelet->Eval(0.85), -1.0);
}

TEST(HaarBasis, UniformRefinement) {
  int ml = 5;

  haar_tree.UniformRefine(ml);
  auto Lambda = haar_tree.NodesPerLevel();
  auto Delta = disc_cons_tree.NodesPerLevel();

  for (int l = 1; l <= ml; ++l) {
    ASSERT_EQ(Lambda[l].size(), std::pow(2, l - 1));
    ASSERT_EQ(Delta[l].size(), std::pow(2, l));

    double h = 1.0 / std::pow(2, l - 1);
    for (auto psi : Lambda[l]) {
      auto [psi_l, psi_n] = psi->labda();
      ASSERT_EQ(psi_l, l);
      ASSERT_EQ(psi->support().front()->Interval().first, h * psi_n);
      ASSERT_EQ(psi->support().back()->Interval().second, h * (psi_n + 1));

      ASSERT_EQ(psi->Eval(h * psi_n + 0.25 * h), 1.0);
      ASSERT_EQ(psi->Eval(h * psi_n + 0.75 * h), -1.0);
    }
  }
}

}  // namespace Time