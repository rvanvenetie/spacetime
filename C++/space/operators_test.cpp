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
  for (int level = 0; level <= max_level; ++level) {
    auto vertex_view = datastructures::TreeView<Vertex>(T.vertex_meta_root);
    vertex_view.UniformRefine(level);

    // Now create the corresponding element tree
    TriangulationView triang(vertex_view);
    auto forward_op = MassPlusScaledStiffnessOperator</*level*/ 2>(triang);
    auto backward_op =
        DirectInverse<MassPlusScaledStiffnessOperator<2>>(triang);
  }
}
