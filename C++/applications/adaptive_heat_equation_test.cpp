#include "adaptive_heat_equation.hpp"

#include "../space/initial_triangulation.hpp"
#include "../spacetime/linear_form.hpp"
#include "../time/basis.hpp"
#include "../tools/linalg.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "problems.hpp"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using spacetime::CreateQuadratureLinearForm;
using spacetime::CreateSumLinearForm;
using spacetime::CreateZeroEvalLinearForm;
using spacetime::GenerateYDelta;
using Time::ortho_tree;
using Time::OrthonormalWaveletFn;
using Time::three_point_tree;
using Time::ThreePointWaveletFn;

namespace applications {

TEST(AdaptiveHeatEquation, CompareToPython) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(1);
  ortho_tree.UniformRefine(1);
  three_point_tree.UniformRefine(1);

  auto [g_lf, u0_lf] = SmoothProblem();

  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      three_point_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
  X_delta.SparseRefine(1);

  auto heat_eq = CreateAdaptiveHeatEquation(std::move(X_delta), std::move(g_lf),
                                            std::move(u0_lf));

  auto result = heat_eq.Solve();
  auto result_nodes = result->Bfs();
  Eigen::VectorXd python_result(result_nodes.size());
  python_result << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.056793956002164525, 0.12413158640093236;
  for (size_t i = 0; i < result_nodes.size(); i++)
    ASSERT_NEAR(result_nodes[i]->value(), python_result[i], 1e-5);

  auto [residual, residual_norm] = heat_eq.Estimate(/*mean_zero*/ false);
  auto residual_nodes = residual->Bfs();
  Eigen::VectorXd python_residual(residual_nodes.size());
  python_residual << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      5.551115123125783e-17, 5.551115123125783e-17, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, -0.01628850213423607, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, -0.004904405566708146, -0.004904405566708167,
      -0.004904405566708146, -0.004904405566708146, -0.006462116191632044,
      -0.006462116191632037, -0.006462116191632044, -0.006462116191632016;
  for (size_t i = 0; i < residual_nodes.size(); i++)
    ASSERT_NEAR(residual_nodes[i]->value(), python_residual[i], 1e-5);

  std::vector<std::pair<int, double>> python_data{{{1, 0.022990516747664815},
                                                   {4, 0.016706324583205395},
                                                   {10, 0.08984241341963645},
                                                   {13, 0.07019458968270276},
                                                   {26, 0.050368099941736744}}};

  auto marked_nodes = heat_eq.Mark();
  ASSERT_EQ(marked_nodes.size(), python_data[0].first);
  ASSERT_NEAR(residual_norm, python_data[0].second, 1e-10);
  heat_eq.Refine(marked_nodes);

  for (size_t iter = 1; iter < 5; iter++) {
    heat_eq.Solve(heat_eq.vec_Xd_out()->ToVectorContainer());
    auto [errors, norm] = heat_eq.Estimate(/*mean_zero*/ false);
    auto marked_nodes = heat_eq.Mark();
    ASSERT_EQ(marked_nodes.size(), python_data[iter].first);
    ASSERT_NEAR(norm, python_data[iter].second, 1e-5);
    heat_eq.Refine(marked_nodes);
  }
}
}  // namespace applications
