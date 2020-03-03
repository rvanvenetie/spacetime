#include "adaptive_heat_equation.hpp"

#include "../space/initial_triangulation.hpp"
#include "../spacetime/linear_form.hpp"
#include "../time/basis.hpp"
#include "../tools/linalg.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using spacetime::GenerateYDelta;
using Time::ortho_tree;
using Time::three_point_tree;

namespace applications {

TEST(AdaptiveHeatEquation, SparseMatVec) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(1);
  ortho_tree.UniformRefine(0);
  three_point_tree.UniformRefine(0);

  auto time_g = [](double t) { return t * t * t; };
  auto space_g = [](double x, double y) { return x * y; };
  auto u0 = [](double x, double y) { return x * y; };

  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      three_point_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
  X_delta.SparseRefine(1);

  AdaptiveHeatEquation heat_eq(
      std::move(X_delta),
      spacetime::CreateQuadratureLinearForm<Time::OrthonormalWaveletFn, 3, 2>(
          time_g, space_g),
      spacetime::CreateZeroEvalLinearForm<Time::ThreePointWaveletFn, 2>(u0));

  auto result = heat_eq.Solve();
  auto residual = heat_eq.Estimate();
}
}  // namespace applications
