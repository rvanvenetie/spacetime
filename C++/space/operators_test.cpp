#include "operators.hpp"

#include "datastructures/multi_tree_view.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"

using namespace space;

constexpr int max_level = 6;

TEST(Operator, InverseTimesForwardOpIsIdentity) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  for (int level = 2; level <= max_level; ++level) {
    auto vertex_view = datastructures::TreeView<Vertex>(T.vertex_meta_root);
    vertex_view.UniformRefine(level);

    // Now create the corresponding element tree
    TriangulationView triang(vertex_view);
    auto forward_op = StiffnessOperator(triang);
    auto backward_op = DirectInverse<StiffnessOperator>(triang);
    for (int i = 0; i < 10; i++) {
      Eigen::VectorXd vec(triang.vertices().size());
      vec.setRandom();
      for (int v = 0; v < triang.vertices().size(); v++)
        if (triang.vertices()[v]->on_domain_boundary) vec[v] = 0.0;
      Eigen::VectorXd vec2 = vec;
      forward_op.Apply(vec2);
      backward_op.Apply(vec2);
      ASSERT_TRUE(vec2.isApprox(vec));

      backward_op.Apply(vec);
      forward_op.Apply(vec);
      ASSERT_TRUE(vec.isApprox(vec2));
    }
  }
}

TEST(Operator, MultigridOperatorDoesSomething) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  for (int level = 2; level <= max_level; ++level) {
    auto vertex_view = datastructures::TreeView<Vertex>(T.vertex_meta_root);
    vertex_view.UniformRefine(level);

    // Now create the corresponding element tree
    TriangulationView triang(vertex_view);
    auto forward_op = MassOperator(triang);
    auto backward_op = MultigridPreconditioner<MassOperator>(triang);
    for (int i = 0; i < 10; i++) {
      Eigen::VectorXd vec(triang.vertices().size());
      vec.setRandom();
      for (int v = 0; v < triang.vertices().size(); v++)
        if (triang.vertices()[v]->on_domain_boundary) vec[v] = 0.0;
      Eigen::VectorXd vec2 = vec;
      forward_op.Apply(vec2);
      backward_op.Apply(vec2);
    }
  }
}
