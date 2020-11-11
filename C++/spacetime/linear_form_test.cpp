#include "linear_form.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "time/bases.hpp"

constexpr int max_level = 7;

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::HaarWaveletFn;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

namespace spacetime {
TEST(LinearForm, Quadrature) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  B.ortho_tree.UniformRefine(max_level);

  auto time_f = [](double t) { return t * t * t; };
  auto space_f = [](double x, double y) { return x * y; };

  auto linform = CreateQuadratureTensorLinearForm<Time::OrthonormalWaveletFn>(
      time_f, space_f, /*time_order*/ 3, /*space_order*/ 2);
  for (int level = 1; level < max_level; level++) {
    auto vec = datastructures::DoubleTreeVector<Time::OrthonormalWaveletFn,
                                                space::HierarchicalBasisFn>(
        B.ortho_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    vec.SparseRefine(level);
    linform->Apply(&vec);
    for (auto phi : vec.Bfs())
      if (!std::get<1>(phi->nodes())->vertex()->on_domain_boundary)
        ASSERT_NE(phi->value(), 0.0);
  }
}

TEST(LinearForm, ZeroEval) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  B.ortho_tree.UniformRefine(max_level);

  auto space_f = [](double x, double y) { return x * y; };

  auto linform = CreateZeroEvalLinearForm<Time::OrthonormalWaveletFn>(
      space_f, /*space_order*/ 2);
  for (int level = 1; level < max_level; level++) {
    auto vec = datastructures::DoubleTreeVector<Time::OrthonormalWaveletFn,
                                                space::HierarchicalBasisFn>(
        B.ortho_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    vec.SparseRefine(level);
    linform->Apply(&vec);
    ASSERT_NE(vec.ToVector().sum(), 0.0);
  }
}

TEST(LinearForm, InterpolationExact) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  B.three_point_tree.UniformRefine(max_level);
  B.ortho_tree.UniformRefine(max_level);
  auto X_delta = std::make_shared<
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  auto Y_delta = DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>(
      B.ortho_tree.meta_root(), T.hierarch_basis_tree.meta_root());

  // Test RHS -- a function that lies in Z_delta on coarest mesh.
  auto time_g = [](double t) { return t; };
  auto space_g = [](double x, double y) { return x + y; };
  auto g = [time_g, space_g](double t, double x, double y) {
    return time_g(t) * space_g(x, y);
  };

  // Compare quadrature with interpolation linform
  InterpolationLinearForm linform_interpol(X_delta, g);
  auto linform_quadrature =
      CreateQuadratureTensorLinearForm<Time::OrthonormalWaveletFn>(
          time_g, space_g, /*time_order*/ 2, /*space_order*/ 2);

  // We shall fix Y_delta, and check whether the interpolation linear form
  // converges.
  Y_delta.UniformRefine(3);
  auto vec_quad = linform_quadrature->Apply(&Y_delta);

  for (int level = 0; level < max_level; level++) {
    X_delta->SparseRefine(level);
    auto vec_interpol = linform_interpol.Apply(&Y_delta);
    ASSERT_TRUE(vec_interpol.isApprox(vec_quad));
  }
}

TEST(LinearForm, InterpolationConverges) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(2 * max_level);
  B.three_point_tree.UniformRefine(max_level);
  B.ortho_tree.UniformRefine(max_level);
  auto X_delta = std::make_shared<
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  auto Y_delta = DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>(
      B.ortho_tree.meta_root(), T.hierarch_basis_tree.meta_root());

  // Test RHS -- a tensor product.
  auto time_g = [](double t) { return t * t * (t - 0.25); };
  auto space_g = [](double x, double y) { return x * x * y; };
  auto g = [time_g, space_g](double t, double x, double y) {
    return time_g(t) * space_g(x, y);
  };

  // Two ways of calculating the L2 inner product with Y_delta:
  // 1) Interpolation of g into Z_delta, and then a mass matrix.
  // 2) Quadrature of g using a tensorized approach.
  InterpolationLinearForm linform_interpol(X_delta, g);
  auto linform_quadrature =
      CreateQuadratureTensorLinearForm<Time::OrthonormalWaveletFn>(
          time_g, space_g, /*time_order*/ 3, /*space_order*/ 3);

  // Check that the interpolation vector converges towards the quadrature
  // veector, which should be exact.
  std::vector<double> error;

  for (int level = 0; level < max_level; level++) {
    X_delta->SparseRefine(2 * level, {2, 1});
    Y_delta.SparseRefine(2 * (level + 1), {2, 1});

    auto vec_quad = linform_quadrature->Apply(&Y_delta);
    auto vec_interpol = linform_interpol.Apply(&Y_delta);
    error.push_back((vec_interpol - vec_quad).norm());
    if (level > 1) ASSERT_GE(error[level - 1] / error[level], 2);
  }
}

}  // namespace spacetime
