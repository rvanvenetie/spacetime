#include "interpolant.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "time/bases.hpp"

constexpr int max_level = 6;

namespace spacetime {

TEST(Interpolant, ProjectsSparse) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.hierarch_tree.UniformRefine(6);

  for (int level = 0; level < 7; level++) {
    auto tree_interpol =
        datastructures::DoubleTreeVector<Time::HierarchicalWaveletFn,
                                         space::HierarchicalBasisFn>(
            B.hierarch_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    tree_interpol.SparseRefine(level);
    // Put some random values into vec_in.
    for (auto nv : tree_interpol.Bfs()) {
      nv->set_random();
    }

    auto vec_tree_interpol = tree_interpol.ToVector();
    auto vec_nodes = tree_interpol.Bfs();
    Interpolate(
        [&](double t, double x, double y) {
          double result = 0;
          for (int i = 0; i < vec_nodes.size(); ++i) {
            result += vec_nodes[i]->node_0()->Eval(t) *
                      vec_nodes[i]->node_1()->Eval(x, y) * vec_tree_interpol[i];
          }
          return result;
        },
        &tree_interpol);

    ASSERT_TRUE(tree_interpol.ToVector().isApprox(vec_tree_interpol));
  }
}

TEST(Interpolant, ProjectBasis) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.hierarch_tree.UniformRefine(6);
  auto Z_delta = datastructures::DoubleTreeVector<Time::HierarchicalWaveletFn,
                                                  space::HierarchicalBasisFn>(
      B.hierarch_tree.meta_root(), T.hierarch_basis_tree.meta_root());

  for (int level = 0; level < 6; level++) {
    Z_delta.SparseRefine(level);

    auto dblnodes = Z_delta.Bfs();
    for (size_t i = 0; i < dblnodes.size(); ++i) {
      Z_delta.Reset();

      // Check that the interpolation of a basis function is exactly the basis
      // fn.
      Interpolate(
          [&](double t, double x, double y) {
            return dblnodes[i]->node_0()->Eval(t) *
                   dblnodes[i]->node_1()->Eval(x, y);
          },
          &Z_delta);
      for (size_t j = 0; j < dblnodes.size(); ++j) {
        if (j != i)
          ASSERT_TRUE(dblnodes[j]->value() == 0);
        else
          ASSERT_TRUE(dblnodes[j]->value() == 1);
      }
    }
  }
}
}  // namespace spacetime
