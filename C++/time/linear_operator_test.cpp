#include "linear_operator.hpp"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <set>
#include <unordered_map>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace Time {
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::Not;

template <typename LinearOperator, typename BasisIn, typename BasisOut>
void CheckMatrixTranspose(const SparseIndices<BasisIn> &indices_in,
                          const SparseIndices<BasisOut> &indices_out) {
  auto op = LinearOperator();
  Eigen::MatrixXd A =
      Eigen::MatrixXd::Zero(indices_out.size(), indices_in.size());
  Eigen::MatrixXd AT =
      Eigen::MatrixXd::Zero(indices_in.size(), indices_out.size());
  std::unordered_map<BasisIn *, int> indices_in_map;
  std::unordered_map<BasisOut *, int> indices_out_map;

  for (int i = 0; i < indices_in.size(); ++i) {
    assert(!indices_in_map.count(indices_in[i]));
    indices_in_map[indices_in[i]] = i;
  }
  for (int i = 0; i < indices_out.size(); ++i) {
    assert(!indices_out_map.count(indices_out[i]));
    indices_out_map[indices_out[i]] = i;
  }

  // Create A.
  for (int i = 0; i < indices_in.size(); ++i) {
    SparseVector<BasisIn> vec{{{indices_in[i], 1.0}}};
    auto op_vec = op.MatVec(vec);
    for (auto [fn, coeff] : op_vec) {
      EXPECT_THAT(coeff, Not(DoubleEq(0)));
      A(indices_out_map[fn], i) = coeff;
    }

    auto op_vec_check = op.MatVec(vec, indices_out);
    for (auto [fn, coeff] : op_vec_check)
      ASSERT_DOUBLE_EQ(coeff, A(indices_out_map[fn], i));
  }

  // Create AT.
  for (int i = 0; i < indices_out.size(); ++i) {
    SparseVector<BasisOut> vec{{{indices_out[i], 1.0}}};
    auto op_vec = op.RMatVec(vec);
    for (auto [fn, coeff] : op_vec) AT(indices_in_map[fn], i) = coeff;

    auto op_vec_check = op.RMatVec(vec, indices_in);
    for (auto [fn, coeff] : op_vec_check)
      ASSERT_DOUBLE_EQ(coeff, AT(indices_in_map[fn], i));
  }

  // Check that they are the same.
  if (!A.transpose().isApprox(AT)) {
    std::cout << A.transpose() - AT << std::endl;
  }
  ASSERT_TRUE(A.transpose().isApprox(AT));
}

TEST(ContLinearScaling, ProlongateEval) {
  // Reset the persistent trees.
  ResetTrees();

  int ml = 15;
  // Now we check what happens when we also refine near the end points.
  three_point_tree.DeepRefine([ml](auto node) {
    return node->is_metaroot() ||
           (node->level() < ml &&
            (node->index() == 0 ||
             node->index() == (1 << (node->level() - 1)) - 1));
  });

  auto Lambda = three_point_tree.NodesPerLevel();
  auto Delta = cont_lin_tree.NodesPerLevel();

  double n_t = 2048;
  for (int l = 0; l < ml; ++l) {
    for (int i = 0; i < Delta[l].size(); ++i) {
      // Prolongate a single hat function.
      SparseVector<ContLinearScalingFn> vec{{{Delta[l][i], 1.0}}};

      auto p_vec = Prolongate<ContLinearScalingFn>()(vec);
      // Check that the functions eval to the same thing.
      for (int x = 0; x < n_t; x++) {
        double t = x * 1.0 / n_t;
        double eval = 0;
        for (auto [phi, coeff] : p_vec) {
          eval += phi->Eval(t) * coeff;
        }
        ASSERT_DOUBLE_EQ(Delta[l][i]->Eval(t), eval);
      }
    }
  }
}

TEST(ContLinearScaling, CheckOperatorTransposes) {
  // Reset the persistent trees.
  ResetTrees();

  int ml = 7;

  three_point_tree.UniformRefine(ml);
  auto Lambda = three_point_tree.NodesPerLevel();
  auto Delta = cont_lin_tree.NodesPerLevel();

  for (int l = 1; l < ml; ++l) {
    CheckMatrixTranspose<Prolongate<ContLinearScalingFn>, ContLinearScalingFn,
                         ContLinearScalingFn>({Delta[l - 1]}, {Delta[l]});
    CheckMatrixTranspose<MassOperator<ContLinearScalingFn, ContLinearScalingFn>,
                         ContLinearScalingFn, ContLinearScalingFn>(
        {Delta[l - 1]}, {Delta[l - 1]});
    CheckMatrixTranspose<
        ZeroEvalOperator<ContLinearScalingFn, ContLinearScalingFn>,
        ContLinearScalingFn, ContLinearScalingFn>({Delta[l - 1]},
                                                  {Delta[l - 1]});
  }
}

TEST(DiscLinearScaling, CheckOperatorTransposes) {
  // Reset the persistent trees.
  ResetTrees();

  int ml = 7;

  ortho_tree.UniformRefine(ml);
  auto Lambda = ortho_tree.NodesPerLevel();
  auto Delta = disc_lin_tree.NodesPerLevel();

  for (int l = 1; l < ml; ++l) {
    CheckMatrixTranspose<Prolongate<DiscLinearScalingFn>, DiscLinearScalingFn,
                         DiscLinearScalingFn>({Delta[l - 1]}, {Delta[l]});
    CheckMatrixTranspose<MassOperator<DiscLinearScalingFn, DiscLinearScalingFn>,
                         DiscLinearScalingFn, DiscLinearScalingFn>(
        {Delta[l - 1]}, {Delta[l - 1]});
    CheckMatrixTranspose<
        ZeroEvalOperator<DiscLinearScalingFn, DiscLinearScalingFn>,
        DiscLinearScalingFn, DiscLinearScalingFn>({Delta[l - 1]},
                                                  {Delta[l - 1]});
  }
}

TEST(DiscContLinearScaling, CheckOperatorTransposes) {
  // Reset the persistent trees.
  ResetTrees();

  int ml = 7;

  three_point_tree.UniformRefine(ml);
  auto Lambda_3pt = three_point_tree.NodesPerLevel();
  auto Delta_3pt = cont_lin_tree.NodesPerLevel();

  ortho_tree.UniformRefine(ml);
  auto Lambda_ortho = ortho_tree.NodesPerLevel();
  auto Delta_ortho = disc_lin_tree.NodesPerLevel();

  for (int l = 1; l < ml; ++l) {
    CheckMatrixTranspose<MassOperator<ContLinearScalingFn, DiscLinearScalingFn>,
                         ContLinearScalingFn, DiscLinearScalingFn>(
        {Delta_3pt[l - 1]}, {Delta_ortho[l - 1]});
    CheckMatrixTranspose<MassOperator<DiscLinearScalingFn, ContLinearScalingFn>,
                         DiscLinearScalingFn, ContLinearScalingFn>(
        {Delta_ortho[l - 1]}, {Delta_3pt[l - 1]});
    CheckMatrixTranspose<
        ZeroEvalOperator<ContLinearScalingFn, DiscLinearScalingFn>,
        ContLinearScalingFn, DiscLinearScalingFn>({Delta_3pt[l - 1]},
                                                  {Delta_ortho[l - 1]});
    CheckMatrixTranspose<
        ZeroEvalOperator<DiscLinearScalingFn, ContLinearScalingFn>,
        DiscLinearScalingFn, ContLinearScalingFn>({Delta_ortho[l - 1]},
                                                  {Delta_3pt[l - 1]});
    CheckMatrixTranspose<
        TransportOperator<ContLinearScalingFn, DiscLinearScalingFn>,
        ContLinearScalingFn, DiscLinearScalingFn>({Delta_3pt[l - 1]},
                                                  {Delta_ortho[l - 1]});
  }
}

}  // namespace Time
