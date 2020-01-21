#include <cmath>
#include <map>
#include <set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "haar_basis.hpp"
#include "orthonormal_basis.hpp"

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

TEST(OrthonormalBasis, UniformRefinement) {
  int ml = 7;

  ortho_tree.UniformRefine(ml);
  auto Lambda = ortho_tree.NodesPerLevel();
  auto Delta = disc_lin_tree.NodesPerLevel();
  for (int l = 1; l <= ml; ++l) {
    ASSERT_EQ(Lambda[l].size(), std::pow(2, l));
    ASSERT_EQ(Delta[l].size(), std::pow(2, l + 1));
    auto h = pow(0.5, l);
    auto h_supp = pow(0.5, l - 1);
    for (auto psi : Lambda[l]) {
      assert(psi != nullptr);
      for (int i = 1; i < psi->children().size(); i++) {
        ASSERT_EQ(psi->children()[i - 1]->index() + 1,
                  psi->children()[i]->index());
      }
      auto [psi_l, psi_n] = psi->labda();
      ASSERT_EQ(psi_l, l);
      ASSERT_EQ(psi->support().front()->Interval().first, h_supp * (psi_n / 2));
      ASSERT_EQ(psi->support().back()->Interval().second,
                h_supp * (psi_n / 2 + 1));

      if (psi_n % 2 == 1) {
        ASSERT_FLOAT_EQ(psi->Eval(2 * h * (psi_n / 2) + 1e-12),
                        sqrt(3) * pow(2.0, (l - 1.0) / 2.0));
        ASSERT_FLOAT_EQ(psi->Eval(h + 2 * h * (psi_n / 2)),
                        -sqrt(3) * pow(2.0, (l - 1.0) / 2.0));
        ASSERT_FLOAT_EQ(psi->Eval(2 * h + 2 * h * (psi_n / 2) - 1e-12),
                        sqrt(3) * pow(2.0, (l - 1.0) / 2.0));
      } else {
        ASSERT_FLOAT_EQ(psi->Eval(2 * h * (psi_n / 2) + 1e-12),
                        pow(2.0, (l - 1.0) / 2.0));
        ASSERT_FLOAT_EQ(psi->Eval(h + 2 * h * (psi_n / 2) - 1e-12),
                        -2 * pow(2.0, (l - 1.0) / 2.0));
        ASSERT_FLOAT_EQ(psi->Eval(h + 2 * h * (psi_n / 2) + 1e-12),
                        2 * pow(2.0, (l - 1.0) / 2.0));
        ASSERT_FLOAT_EQ(psi->Eval(2 * h + 2 * h * (psi_n / 2) - 1e-12),
                        -pow(2.0, (l - 1.0) / 2.0));
      }
    }
  }
}

TEST(OrthonormalBasis, LocalRefinement) {
  int ml = 15;

  // Reset the persistent wavelet trees.
  disc_lin_tree = datastructures::Tree<DiscLinearScalingFn>();
  ortho_tree = datastructures::Tree<OrthonormalWaveletFn>();

  // Refine towards t=0.
  ortho_tree.DeepRefine([ml](auto node) {
    return node->is_metaroot() || (node->level() < ml && node->index() == 0);
  });
  auto Lambda = ortho_tree.NodesPerLevel();
  auto Delta = disc_lin_tree.NodesPerLevel();

  ASSERT_EQ(Lambda[0].size(), 2);
  ASSERT_EQ(Lambda[1].size(), 2);
  for (int l = 2; l <= ml; ++l) {
    ASSERT_EQ(Lambda[l].size(), 4);
    ASSERT_EQ(Delta[l].size(), 8);
    ASSERT_EQ(Lambda[l][0]->labda(), std::pair(l, 0));
  }
}

}  // namespace Time
