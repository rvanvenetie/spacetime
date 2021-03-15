#include "linear_operator.hpp"

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

    auto vec_tree = tree_interpol.ToVector();
    auto vec_nodes = tree_interpol.Bfs();

    auto vec_tree_interpol = Interpolate(
        [&](double t, double x, double y) {
          double result = 0;
          for (int i = 0; i < vec_nodes.size(); ++i) {
            result += vec_nodes[i]->node_0()->Eval(t) *
                      vec_nodes[i]->node_1()->Eval(x, y) * vec_tree[i];
          }
          return result;
        },
        tree_interpol);
    tree_interpol.FromVectorContainer(vec_tree_interpol);

    ASSERT_TRUE(vec_tree.isApprox(tree_interpol.ToVector()));
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
      auto vec = Interpolate(
          [&](double t, double x, double y) {
            return dblnodes[i]->node_0()->Eval(t) *
                   dblnodes[i]->node_1()->Eval(x, y);
          },
          Z_delta);
      Z_delta.FromVectorContainer(vec);
      for (size_t j = 0; j < dblnodes.size(); ++j) {
        if (j != i)
          ASSERT_TRUE(dblnodes[j]->value() == 0);
        else
          ASSERT_TRUE(dblnodes[j]->value() == 1);
      }
    }
  }
}

TEST(Trace, Works) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.ortho_tree.UniformRefine(6);
  B.three_point_tree.UniformRefine(6);

  for (int level = 1; level < 6; level++) {
    auto vec_X = datastructures::DoubleTreeVector<Time::ThreePointWaveletFn,
                                                  space::HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    vec_X.SparseRefine(level);

    // Put some random values into vec_X.
    for (auto nv : vec_X.Bfs()) {
      if (std::get<1>(nv->nodes())->on_domain_boundary()) continue;
      nv->set_random();
    }

    // Calculate trace of the operator.
    auto vec_X_0 = Trace(0, vec_X);

    // Create evaluation lambdas.
    auto vec_X_nodes = vec_X.Bfs();
    auto vec_X_0_nodes = vec_X_0.Bfs();
    auto eval_vec_X = [&](double x, double y) {
      double result = 0;
      for (int i = 0; i < vec_X_nodes.size(); ++i) {
        result += vec_X_nodes[i]->node_0()->Eval(0) *
                  vec_X_nodes[i]->node_1()->Eval(x, y) *
                  vec_X_nodes[i]->value();
      }
      return result;
    };
    auto eval_vec_X_0 = [&](double x, double y) {
      double result = 0;
      for (int i = 0; i < vec_X_0_nodes.size(); ++i) {
        result +=
            vec_X_0_nodes[i]->node()->Eval(x, y) * vec_X_0_nodes[i]->value();
      }
      return result;
    };

    // Evaluate on some points in the plane
    for (double x : {0.0, 0.12341, 0.23453, 0.5, 0.5234, 1.0})
      for (double y : {0.0, 0.12341, 0.23453, 0.5, 0.5234, 1.0}) {
        ASSERT_NEAR(eval_vec_X_0(x, y), eval_vec_X(x, y), 1e-12);
      }
  }
}

}  // namespace spacetime
