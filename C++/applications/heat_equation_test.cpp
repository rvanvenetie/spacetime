#include "heat_equation.hpp"

#include "../space/initial_triangulation.hpp"
#include "../time/basis.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using spacetime::GenerateYDelta;
using Time::ortho_tree;
using Time::three_point_tree;

namespace applications {

TEST(HeatEquation, SparseMatVec) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  ortho_tree.UniformRefine(6);
  three_point_tree.UniformRefine(6);

  for (int level = 1; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.SparseRefine(level);

    HeatEquation heat_eq(X_delta);

    // Generate some random rhs.
    for (auto nv : heat_eq.vec_X_in()->Bfs()) {
      if (nv->node_1()->on_domain_boundary()) continue;
      nv->set_value(((double)std::rand()) / RAND_MAX);
    }
    for (auto nv : heat_eq.vec_Y_in()->Bfs()) {
      if (nv->node_1()->on_domain_boundary()) continue;
      nv->set_value(((double)std::rand()) / RAND_MAX);
    }

    // Apply the block matrix :-).
    heat_eq.BlockMat()->Apply();

    // Validate the result.
    ValidateVector(*heat_eq.vec_X_out());
    ValidateVector(*heat_eq.vec_Y_out());
  }
}
}  // namespace applications
