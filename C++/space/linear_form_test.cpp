#include "linear_form.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"

using namespace datastructures;

constexpr int max_level = 7;

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

namespace space {
TEST(LinearForm, Quadrature) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);

  auto f = [](double x, double y) { return x * y; };
  auto lf = LinearForm(f, /* apply_quadrature*/ true, /*order*/ 2,
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

TEST(LinearForm, InterpolationExact) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);

  auto f = [](double x, double y) { return x + y; };
  for (auto dirichlet_boundary : {false, true}) {
    auto lf_quad = LinearForm(f, /* apply_quadrature*/ true, /*order*/ 1,
                              /* dirichlet_boundary */ dirichlet_boundary);
    auto lf_interpol = LinearForm(f, /* apply_quadrature*/ false, /*order*/ 0,
                                  /* dirichlet_boundary */ dirichlet_boundary);
    for (int j = 0; j < 20; ++j) {
      TreeVector<HierarchicalBasisFn> vec(T.hierarch_basis_meta_root);
      vec.DeepRefine(
          /* call_filter */ [](auto&& nv) {
            return nv->level() <= 0 || bsd_rnd() % 3 != 0;
          });

      lf_quad.Apply(vec.root());
      auto vec_quad = vec.ToVector();
      lf_interpol.Apply(vec.root());
      auto vec_interpol = vec.ToVector();

      ASSERT_TRUE(vec_quad.isApprox(vec_interpol));
    }
  }
}

TEST(LinearForm, InterpolationConverges) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);

  auto f = [](double x, double y) { return x * x * y; };
  for (auto dirichlet_boundary : {false, true}) {
    auto lf_quad = LinearForm(f, /* apply_quadrature*/ true, /*order*/ 3,
                              /* dirichlet_boundary */ dirichlet_boundary);
    auto lf_interpol = LinearForm(f, /* apply_quadrature*/ false, /*order*/ 0,
                                  /* dirichlet_boundary */ dirichlet_boundary);

    std::vector<double> error;
    for (int level = 0; level <= max_level; level++) {
      TreeVector<HierarchicalBasisFn> vec(T.hierarch_basis_meta_root);
      vec.UniformRefine(level);

      lf_quad.Apply(vec.root());
      auto vec_quad = vec.ToVector();
      lf_interpol.Apply(vec.root());
      auto vec_interpol = vec.ToVector();

      error.push_back((vec_interpol - vec_quad).norm());
      if (level > 1) ASSERT_GE(error[level - 1] / error[level], 1.5);
    }
  }
}
};  // namespace space
