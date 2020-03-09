#include "adaptive_heat_equation.hpp"

#include "../space/initial_triangulation.hpp"
#include "../spacetime/linear_form.hpp"
#include "../time/basis.hpp"
#include "../tools/linalg.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

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

TEST(AdaptiveHeatEquation, SparseMatVec) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(1);
  ortho_tree.UniformRefine(1);
  three_point_tree.UniformRefine(1);

  auto time_g1 = [](double t) { return -2 * (1 + t * t); };
  auto space_g1 = [](double x, double y) { return (x - 1) * x + (y - 1) * y; };
  auto time_g2 = [](double t) { return 2 * t; };
  auto space_g2 = [](double x, double y) { return (x - 1) * x * (y - 1) * y; };
  auto u0 = [](double x, double y) { return (1 - x) * x * (1 - y) * y; };

  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      three_point_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
  X_delta.SparseRefine(1);

  AdaptiveHeatEquation heat_eq(
      std::move(X_delta),
      CreateSumLinearForm<OrthonormalWaveletFn>(
          CreateQuadratureLinearForm<OrthonormalWaveletFn, 2, 2>(time_g1,
                                                                 space_g1),
          CreateQuadratureLinearForm<OrthonormalWaveletFn, 1, 4>(time_g2,
                                                                 space_g2)),
      CreateZeroEvalLinearForm<ThreePointWaveletFn, 4>(u0));

  auto result = heat_eq.Solve();
  auto nodes = heat_eq.vec_Xd_out().Bfs();
  Eigen::VectorXd python_result(nodes.size());
  python_result << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.056793956002164525, 0.12413158640093236;
  heat_eq.vec_Xd_out().FromVectorContainer(result);
  for (size_t i = 0; i < nodes.size(); i++)
    ASSERT_NEAR(nodes[i]->value(), python_result[i], 1e-5);

  auto residual = heat_eq.Estimate(/*mean_zero*/ false);
  auto residual_nodes = residual.Bfs();

  Eigen::VectorXd python_residual(residual_nodes.size());
  python_residual << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      5.551115123125783e-17, 5.551115123125783e-17, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, -0.01628850213423607, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, -0.004904405566708146, -0.004904405566708167,
      -0.004904405566708146, -0.004904405566708146, -0.006462116191632044,
      -0.006462116191632037, -0.006462116191632044, -0.006462116191632016;
  for (size_t i = 0; i < residual_nodes.size(); i++)
    ASSERT_NEAR(residual_nodes[i]->value(), python_residual[i], 1e-5);
}
}  // namespace applications
