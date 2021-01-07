#include "linear_operator.hpp"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <set>
#include <unordered_map>

#include "bases.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "integration.hpp"

namespace Time {
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::Not;

template <typename LinearOperator, typename BasisIn, typename BasisOut>
void CheckMatrixTranspose(const SparseIndices<BasisIn> &indices_in,
                          const SparseIndices<BasisOut> &indices_out) {
  auto op = LinearOperator();
  Eigen::MatrixXd A = op.ToMatrix(indices_in, indices_out);
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

  // Check A.
  for (int i = 0; i < indices_in.size(); ++i) {
    SparseVector<BasisIn> vec{{{indices_in[i], 1.0}}};
    auto op_vec = op.MatVec(vec);
    for (auto [fn, coeff] : op_vec) {
      EXPECT_THAT(coeff, Not(DoubleEq(0)));
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
  ASSERT_TRUE(A.transpose().isApprox(AT));
}

template <typename LinearOperator, typename BasisIn, typename BasisOut>
void CheckMatrixQuadrature(const SparseIndices<BasisIn> &indices_in,
                           bool deriv_in,
                           const SparseIndices<BasisOut> &indices_out,
                           bool deriv_out) {
  auto mat = LinearOperator().ToMatrix(indices_in, indices_out);
  for (int j = 0; j < indices_in.size(); ++j)
    for (int i = 0; i < indices_out.size(); ++i) {
      auto psi_j = indices_in[j];
      auto phi_i = indices_out[i];

      double ip = 0;
      auto eval = [psi_j, deriv_in, phi_i, deriv_out](double t) {
        return psi_j->Eval(t, deriv_in) * phi_i->Eval(t, deriv_out);
      };
      for (auto elem : phi_i->support())
        ip += Integrate(eval, *elem, /*degree*/ 3);

      ASSERT_NEAR(mat(i, j), ip, 1e-10);
    }
}

TEST(ContLinearScaling, ProlongateEval) {
  // Reset the persistent trees.
  Bases B;

  int ml = 64;
  // Now we check what happens when we also refine near the end points.
  B.three_point_tree.DeepRefine([ml](auto node) {
    return node->is_metaroot() ||
           (node->level() < ml &&
            (node->index() == 0 ||
             node->index() == (1LL << (node->level() - 1)) - 1));
  });

  auto Lambda = B.three_point_tree.NodesPerLevel();
  auto Delta = B.cont_lin_tree.NodesPerLevel();

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

TEST(ContLinearScaling, CheckMatrixTransposes) {
  // Reset the persistent trees.
  Bases B;

  int ml = 7;

  B.three_point_tree.UniformRefine(ml);
  auto Lambda = B.three_point_tree.NodesPerLevel();
  auto Delta = B.cont_lin_tree.NodesPerLevel();

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

TEST(ContLinearScaling, MatrixQuadrature) {
  // Reset the persistent trees.
  Bases B;

  int ml = 7;

  B.three_point_tree.UniformRefine(ml);
  auto Lambda = B.three_point_tree.NodesPerLevel();
  auto Delta = B.cont_lin_tree.NodesPerLevel();

  for (int l = 0; l < ml; ++l) {
    CheckMatrixQuadrature<
        MassOperator<ContLinearScalingFn, ContLinearScalingFn>,
        ContLinearScalingFn, ContLinearScalingFn>({Delta[l]}, false, {Delta[l]},
                                                  false);
  }
}

TEST(DiscLinearScaling, CheckMatrixTransposes) {
  // Reset the persistent trees.
  Bases B;

  int ml = 7;

  B.ortho_tree.UniformRefine(ml);
  auto Lambda = B.ortho_tree.NodesPerLevel();
  auto Delta = B.disc_lin_tree.NodesPerLevel();

  for (int l = 1; l < ml; ++l) {
    std::cout << "Prolongation" << std::endl;
    CheckMatrixTranspose<Prolongate<DiscLinearScalingFn>, DiscLinearScalingFn,
                         DiscLinearScalingFn>({Delta[l - 1]}, {Delta[l]});
    std::cout << "MassOperator" << std::endl;
    CheckMatrixTranspose<MassOperator<DiscLinearScalingFn, DiscLinearScalingFn>,
                         DiscLinearScalingFn, DiscLinearScalingFn>(
        {Delta[l - 1]}, {Delta[l - 1]});
    std::cout << "ZeroEvalOperator" << std::endl;
    CheckMatrixTranspose<
        ZeroEvalOperator<DiscLinearScalingFn, DiscLinearScalingFn>,
        DiscLinearScalingFn, DiscLinearScalingFn>({Delta[l - 1]},
                                                  {Delta[l - 1]});
  }
}

TEST(DiscLinearScaling, MatrixQuadrature) {
  // Reset the persistent trees.
  Bases B;

  int ml = 7;

  B.ortho_tree.UniformRefine(ml);
  auto Lambda = B.ortho_tree.NodesPerLevel();
  auto Delta = B.disc_lin_tree.NodesPerLevel();

  for (int l = 0; l < ml; ++l) {
    CheckMatrixQuadrature<
        MassOperator<DiscLinearScalingFn, DiscLinearScalingFn>,
        DiscLinearScalingFn, DiscLinearScalingFn>({Delta[l]}, false, {Delta[l]},
                                                  false);
  }
}

TEST(DiscContLinearScaling, CheckMatrixTransposes) {
  // Reset the persistent trees.
  Bases B;

  int ml = 7;

  B.three_point_tree.UniformRefine(ml);
  auto Lambda_3pt = B.three_point_tree.NodesPerLevel();
  auto Delta_3pt = B.cont_lin_tree.NodesPerLevel();

  B.ortho_tree.UniformRefine(ml);
  auto Lambda_ortho = B.ortho_tree.NodesPerLevel();
  auto Delta_ortho = B.disc_lin_tree.NodesPerLevel();

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

TEST(DiscContLinearScaling, ZeroEvalWorks) {
  // Reset the persistent trees.
  Bases B;

  int ml = 7;

  B.three_point_tree.UniformRefine(ml);
  auto Lambda_3pt = B.three_point_tree.NodesPerLevel();
  auto Delta_3pt = B.cont_lin_tree.NodesPerLevel();

  B.ortho_tree.UniformRefine(ml);
  auto Lambda_ortho = B.ortho_tree.NodesPerLevel();
  auto Delta_ortho = B.disc_lin_tree.NodesPerLevel();

  for (int l = 0; l < ml; ++l) {
    auto mat =
        ZeroEvalOperator<ContLinearScalingFn, DiscLinearScalingFn>().ToMatrix(
            {Delta_3pt[l]}, {Delta_ortho[l]});
    for (int j = 0; j < Delta_3pt[l].size(); ++j)
      for (int i = 0; i < Delta_ortho[l].size(); ++i)
        ASSERT_NEAR(mat(i, j),
                    Delta_3pt[l][j]->Eval(0.0) * Delta_ortho[l][i]->Eval(0.0),
                    1e-10);
  }
}
TEST(DiscContLinearScaling, MatrixQuadrature) {
  // Reset the persistent trees.
  Bases B;

  int ml = 7;

  B.three_point_tree.UniformRefine(ml);
  auto Lambda_3pt = B.three_point_tree.NodesPerLevel();
  auto Delta_3pt = B.cont_lin_tree.NodesPerLevel();

  B.ortho_tree.UniformRefine(ml);
  auto Lambda_ortho = B.ortho_tree.NodesPerLevel();
  auto Delta_ortho = B.disc_lin_tree.NodesPerLevel();

  for (int l = 0; l < ml; ++l) {
    CheckMatrixQuadrature<
        MassOperator<ContLinearScalingFn, DiscLinearScalingFn>,
        ContLinearScalingFn, DiscLinearScalingFn>({Delta_3pt[l]}, false,
                                                  {Delta_ortho[l]}, false);
    CheckMatrixQuadrature<
        MassOperator<DiscLinearScalingFn, ContLinearScalingFn>,
        DiscLinearScalingFn, ContLinearScalingFn>({Delta_ortho[l]}, false,
                                                  {Delta_3pt[l]}, false);
    bool matrices_are_transposes =
        MassOperator<ContLinearScalingFn, DiscLinearScalingFn>()
            .ToMatrix({Delta_3pt[l]}, {Delta_ortho[l]})
            .transpose()
            .isApprox(MassOperator<DiscLinearScalingFn, ContLinearScalingFn>()
                          .ToMatrix({Delta_ortho[l]}, {Delta_3pt[l]}));
    ASSERT_TRUE(matrices_are_transposes);

    CheckMatrixQuadrature<
        TransportOperator<ContLinearScalingFn, DiscLinearScalingFn>,
        ContLinearScalingFn, DiscLinearScalingFn>({Delta_3pt[l]}, true,
                                                  {Delta_ortho[l]}, false);
  }
}

}  // namespace Time
