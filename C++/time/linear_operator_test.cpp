#include "linear_operator.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <set>
#include <unordered_map>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace Time {
using ::testing::ElementsAre;

template <typename LinearOperator, typename basis_in, typename basis_out>
void CheckMatrixTranspose(const LinearOperator &op,
                          std::vector<basis_in *> indices_in,
                          std::vector<basis_out *> indices_out) {
  Eigen::MatrixXd A =
      Eigen::MatrixXd::Zero(indices_out.size(), indices_in.size());
  Eigen::MatrixXd AT =
      Eigen::MatrixXd::Zero(indices_in.size(), indices_out.size());
  std::unordered_map<basis_in *, int> indices_in_map;
  std::unordered_map<basis_out *, int> indices_out_map;

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
    SparseVector<basis_in> vec(std::vector{std::pair{indices_in[i], 1.0}});
    auto op_vec = op.MatVec(vec);
    for (auto [fn, coeff] : op_vec) A(indices_out_map[fn], i) = coeff;

    auto op_vec_check = op.MatVec(vec, indices_out);
    for (auto [fn, coeff] : op_vec_check)
      ASSERT_DOUBLE_EQ(coeff, A(indices_out_map[fn], i));
  }

  // Create AT
  for (int i = 0; i < indices_out.size(); ++i) {
    SparseVector<basis_out> vec(std::vector{std::pair{indices_out[i], 1.0}});
    auto op_vec = op.RMatVec(vec, indices_in);
    for (auto [fn, coeff] : op_vec) AT(indices_in_map[fn], i) = coeff;

    auto op_vec_check = op.MatVec(vec, indices_in);
    for (auto [fn, coeff] : op_vec_check)
      ASSERT_DOUBLE_EQ(coeff, A(indices_in_map[fn], i));
  }

  // Check that they are the same.
  ASSERT_DOUBLE_EQ((A.transpose() - AT).cwiseAbs().maxCoeff(), 0);
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
      SparseVector<ContLinearScalingFn> vec(
          std::vector{std::pair{Delta[l][i], 1.0}});

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

TEST(ContLinearScaling, ProlongateMatrix) {
  // Reset the persistent trees.
  ResetTrees();

  int ml = 7;

  three_point_tree.UniformRefine(ml);
  auto Lambda = three_point_tree.NodesPerLevel();
  auto Delta = cont_lin_tree.NodesPerLevel();

  for (int l = 1; l < ml; ++l) {
    CheckMatrixTranspose(Prolongate<ContLinearScalingFn>(), Delta[l - 1],
                         Delta[l]);
  }
}

TEST(ContLinearScaling, MassMatrix) {
  // Reset the persistent trees.
  ResetTrees();

  int ml = 7;

  three_point_tree.UniformRefine(ml);
  auto Lambda = three_point_tree.NodesPerLevel();
  auto Delta = cont_lin_tree.NodesPerLevel();

  for (int l = 1; l < ml; ++l) {
    CheckMatrixTranspose(
        MassOperator<ContLinearScalingFn, ContLinearScalingFn>(), Delta[l - 1],
        Delta[l - 1]);
  }
}

}  // namespace Time
