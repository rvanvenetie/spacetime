#include "interpolant.hpp"

#include "datastructures/multi_tree_vector.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"

using namespace datastructures;

constexpr int max_level = 5;

namespace space {
TEST(Interpolant, Projection) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);

  for (int level = 0; level <= max_level; ++level) {
    auto tree_vec = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
    tree_vec.UniformRefine(level);

    // Put some random values into tree_vec.
    for (auto nv : tree_vec.Bfs()) nv->set_random();

    auto vec = tree_vec.ToVector();
    auto vec_nodes = tree_vec.Bfs();

    Interpolate(
        [&](double x, double y) {
          double result = 0;
          for (int i = 0; i < vec_nodes.size(); ++i) {
            result += vec_nodes[i]->node()->Eval(x, y) * vec[i];
          }
          return result;
        },
        tree_vec.root());
    ASSERT_TRUE(tree_vec.ToVector().isApprox(vec));
  }
}
};  // namespace space
