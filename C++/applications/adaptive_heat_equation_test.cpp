#include "adaptive_heat_equation.hpp"

#include "../space/initial_triangulation.hpp"
#include "../time/basis.hpp"
#include "../tools/linalg.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "problems.hpp"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using spacetime::CreateQuadratureLinearForm;
using spacetime::CreateZeroEvalLinearForm;
using spacetime::GenerateYDelta;
using spacetime::LinearForm;
using spacetime::SumLinearForm;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

namespace applications {

TEST(AdaptiveHeatEquation, CompareToPython) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(1);
  B.ortho_tree.UniformRefine(1);
  B.three_point_tree.UniformRefine(1);

  auto time_g1 = [](double t) { return -2 * (1 + t * t); };
  auto space_g1 = [](double x, double y) { return (x - 1) * x + (y - 1) * y; };
  auto time_g2 = [](double t) { return 2 * t; };
  auto space_g2 = [](double x, double y) { return (x - 1) * x * (y - 1) * y; };
  auto u0 = [](double x, double y) { return (1 - x) * x * (1 - y) * y; };

  auto vec_Xd = std::make_shared<
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  vec_Xd->SparseRefine(1);

  auto [g_lf, u0_lf] = SmoothProblem();
  AdaptiveHeatEquationOptions opts = {.estimate_mean_zero_ = false};
  opts.P_X_alpha_ = 0.35;
  AdaptiveHeatEquation heat_eq(vec_Xd, std::move(g_lf), std::move(u0_lf), opts);

  std::vector<size_t> python_pcg_iters{2, 3, 5, 5, 5};
  auto [u, pcg_data] = heat_eq.Solve();
  vec_Xd->FromVectorContainer(u);
  auto u_nodes = vec_Xd->Bfs();
  Eigen::VectorXd python_u(u_nodes.size());
  python_u << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.056793956002164525, 0.12413158640093236;
  for (size_t i = 0; i < u_nodes.size(); i++)
    ASSERT_NEAR(u_nodes[i]->value(), python_u[i], 1e-5);
  ASSERT_NEAR(pcg_data.iterations, python_pcg_iters[0], 1);

  auto [residual, residual_norm] = heat_eq.Estimate(u);
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

  std::vector<std::pair<int, double>> python_mark_data{
      {{1, 0.022990516747664815},
       {4, 0.016706324583205395},
       {10, 0.08984241341963645},
       {13, 0.07019458968270276},
       {26, 0.050368099941736744}}};

  auto marked_nodes = heat_eq.Mark(residual);
  ASSERT_EQ(marked_nodes.size(), python_mark_data[0].first);
  ASSERT_NEAR(residual_norm, python_mark_data[0].second, 1e-10);

  vec_Xd->FromVectorContainer(u);
  heat_eq.Refine(marked_nodes);

  for (size_t iter = 1; iter < 5; iter++) {
    auto [u, pcg_data] = heat_eq.Solve(vec_Xd->ToVectorContainer());
    auto [residual, norm] = heat_eq.Estimate(u);
    auto marked_nodes = heat_eq.Mark(residual);
    ASSERT_EQ(marked_nodes.size(), python_mark_data[iter].first);
    ASSERT_NEAR(norm, python_mark_data[iter].second, 1e-5);
    ASSERT_NEAR(pcg_data.iterations, python_pcg_iters[iter], 1);

    vec_Xd->FromVectorContainer(u);
    heat_eq.Refine(marked_nodes);
  }
}
}  // namespace applications
