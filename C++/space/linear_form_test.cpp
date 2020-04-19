#include "linear_form.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"

using namespace datastructures;

constexpr int max_level = 2;

namespace space {
TEST(LinearForm, Quadrature) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);

  auto f = [](double x, double y) { return x * y; };
  auto lf = LinearForm(std::make_unique<QuadratureFunctional>(f, /*order*/ 2),
                       /* dirichlet_boundary */ false);
  Eigen::VectorXd prev_vec;
  for (int level = 0; level <= max_level; ++level) {
    auto vec = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
    vec.UniformRefine(level);
    lf.Apply(vec.root());
    for (auto nv : vec.Bfs()) ASSERT_NE(nv->value(), 0.0);
    auto eigen_vec = vec.ToVector();
    if (level > 0)
      for (int i = 0; i < prev_vec.size(); i++)
        ASSERT_NEAR(eigen_vec[i], prev_vec[i], 1e-10);
    prev_vec = eigen_vec;
  }
}
};  // namespace space
