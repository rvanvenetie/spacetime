#include <cmath>
#include <map>
#include <set>

#include "bases.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace Time {
using ::testing::ElementsAre;

TEST(HaarBasis, functions) {
  // Reset the persistent trees.
  Bases B;

  ASSERT_EQ(B.disc_cons_tree.meta_root()->children().size(), 1);
  ASSERT_EQ(B.haar_tree.meta_root()->children().size(), 1);

  auto mother_scaling = B.disc_cons_tree.meta_root()->children()[0];
  ASSERT_EQ(mother_scaling->labda(), std::make_pair(0, 0LL));
  ASSERT_EQ(mother_scaling->Eval(0.25), 1.0);
  ASSERT_EQ(mother_scaling->Eval(0.85), 1.0);

  auto mother_wavelet = B.haar_tree.meta_root()->children()[0];
  mother_wavelet->Refine();
  ASSERT_EQ(mother_wavelet->children().size(), 1);
  mother_wavelet = mother_wavelet->children()[0];
  ASSERT_EQ(mother_wavelet->Eval(0.25), 1.0);
  ASSERT_EQ(mother_wavelet->Eval(0.85), -1.0);
}

TEST(HaarBasis, UniformRefinement) {
  // Reset the persistent trees.
  Bases B;
  int ml = 5;

  B.haar_tree.UniformRefine(ml);
  auto Lambda = B.haar_tree.NodesPerLevel();
  auto Delta = B.disc_cons_tree.NodesPerLevel();

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
  // Reset the persistent trees.
  Bases B;

  int ml = 7;

  B.ortho_tree.UniformRefine(ml);
  auto Lambda = B.ortho_tree.NodesPerLevel();
  auto Delta = B.disc_lin_tree.NodesPerLevel();
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
  // Reset the persistent trees.
  Bases B;

  int ml = 15;

  // Refine towards t=0.
  B.ortho_tree.DeepRefine([ml](auto node) {
    return node->is_metaroot() || (node->level() < ml && node->index() == 0);
  });
  auto Lambda = B.ortho_tree.NodesPerLevel();
  auto Delta = B.disc_lin_tree.NodesPerLevel();

  ASSERT_EQ(Lambda[0].size(), 2);
  ASSERT_EQ(Lambda[1].size(), 2);
  for (int l = 2; l <= ml; ++l) {
    ASSERT_EQ(Lambda[l].size(), 4);
    ASSERT_EQ(Delta[l].size(), 8);
    ASSERT_EQ(Lambda[l][0]->labda(), std::pair(l, 0LL));
  }
}

TEST(ThreePointBasis, UniformRefinement) {
  // Reset the persistent trees.
  Bases B;

  int ml = 7;

  B.three_point_tree.UniformRefine(ml);

  auto mother_scalings = B.three_point_tree.meta_root()->children();
  ASSERT_EQ(mother_scalings[0]->Eval(0), 1.0);
  ASSERT_EQ(mother_scalings[0]->Eval(1), 0);
  ASSERT_EQ(mother_scalings[1]->Eval(0), 0);
  ASSERT_EQ(mother_scalings[1]->Eval(1), 1);

  auto Lambda = B.three_point_tree.NodesPerLevel();
  auto Delta = B.cont_lin_tree.NodesPerLevel();

  ASSERT_EQ(B.three_point_tree.meta_root()->children().size(), 2);
  ASSERT_EQ(B.cont_lin_tree.meta_root()->children().size(), 2);

  for (int l = 1; l <= ml; ++l) {
    ASSERT_EQ(Lambda[l].size(), std::pow(2, l - 1));
    ASSERT_EQ(Delta[l].size(), std::pow(2, l) + 1);

    double h = 1.0 / std::pow(2, l);
    for (auto psi : Lambda[l]) {
      auto [psi_l, psi_n] = psi->labda();
      ASSERT_EQ(psi_l, l);
      ASSERT_EQ(psi->Eval(h * (2 * psi_n + 1)), pow(2, l / 2.0));

      if (psi_n > 0) {
        ASSERT_EQ(psi->Eval(h * 2 * psi_n), -0.5 * pow(2, l / 2.0));
        ASSERT_EQ(psi->Eval(0), 0);
        ASSERT_EQ(psi->Eval(0.0001), 0);
        ASSERT_EQ(psi->support().front()->Interval().first,
                  h * (2 * psi_n - 1));
      }
      if (psi_n < pow(2, l - 1) - 1) {
        ASSERT_EQ(psi->Eval(h * (2 * psi_n + 2)), -0.5 * pow(2, l / 2.0));
        ASSERT_EQ(psi->support().back()->Interval().second,
                  h * (2 * psi_n + 3));
      }

      if (psi_n == 0) {
        ASSERT_EQ(psi->Eval(0), -pow(2, l / 2.0));
        ASSERT_EQ(psi->support().front()->Interval().first, 0);
      }

      if (psi_n == pow(2, l - 1) - 1) {
        ASSERT_EQ(psi->Eval(1), -pow(2, l / 2.0));
        ASSERT_EQ(psi->support().back()->Interval().second, 1);
      }
    }
  }
}

TEST(ThreePointBasis, LocalRefinement) {
  // Reset the persistent trees.
  Bases B;

  int ml = 34;
  // First check what happens when we only refine near the origin.
  B.three_point_tree.DeepRefine([ml](auto node) {
    return node->is_metaroot() || (node->level() < ml && node->index() == 0);
  });
  auto Lambda = B.three_point_tree.NodesPerLevel();
  auto Delta = B.cont_lin_tree.NodesPerLevel();

  ASSERT_EQ(Lambda[0].size(), 2);
  ASSERT_EQ(Lambda[1].size(), 1);
  for (int l = 2; l <= ml; ++l) {
    ASSERT_EQ(Lambda[l].size(), 2);
    ASSERT_EQ(Delta[l].size(), 5);
    ASSERT_EQ(Lambda[l][0]->labda(), std::pair(l, 0LL));
    ASSERT_EQ(Lambda[l][1]->labda(), std::pair(l, 1LL));
  }

  // Now we check what happens when we also refine near the end points.
  B.three_point_tree.DeepRefine([ml](auto node) {
    return node->is_metaroot() ||
           (node->level() < ml &&
            (node->index() == 0 ||
             node->index() == (1LL << (node->level() - 1)) - 1));
  });
  Lambda = B.three_point_tree.NodesPerLevel();
  Delta = B.cont_lin_tree.NodesPerLevel();
  for (int l = 4; l <= ml; ++l) {
    ASSERT_EQ(Lambda[l].size(), 4);
    ASSERT_EQ(Delta[l].size(), 10);
    ASSERT_EQ(Lambda[l][0]->labda(), std::pair(l, 0LL));
    ASSERT_EQ(Lambda[l][1]->labda(), std::pair(l, 1LL));
    ASSERT_EQ(Lambda[l][2]->labda(), std::pair(l, (1LL << (l - 1)) - 2));
    ASSERT_EQ(Lambda[l][3]->labda(), std::pair(l, (1LL << (l - 1)) - 1));
  }
}

}  // namespace Time
