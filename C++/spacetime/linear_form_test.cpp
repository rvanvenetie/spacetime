#include "linear_form.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "time/bases.hpp"

constexpr int max_level = 6;

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

TEST(LinearForm, Interpolation) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  B.three_point_tree.UniformRefine(max_level);

  auto g = [](double t, double x, double y) {
    return t * t * t * x * (x - 1) * y * (y - 1);
  };
  auto time_g = [](double t) { return t * t * t; };
  auto space_g = [](double x, double y) { return x * (x - 1) * y * (y - 1); };

  auto X_delta = std::make_shared<
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  auto Z_delta =
      DoubleTreeVector<Time::HierarchicalWaveletFn, HierarchicalBasisFn>(
          B.hierarch_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  InterpolationLinearForm linform_interpol(X_delta, g);
  auto linform_quadrature =
      CreateQuadratureTensorLinearForm<Time::OrthonormalWaveletFn>(
          time_g, space_g, /*time_order*/ 3, /*space_order*/ 2);
  for (int level = 2; level < max_level; level++) {
    X_delta->SparseRefine(level);
    GenerateZDelta(*X_delta, &Z_delta);

    auto vec_Y = GenerateYDelta<DoubleTreeVector>(*X_delta);
    auto vec_interpol = linform_interpol.Apply(&vec_Y);
    auto vec_quad = linform_quadrature->Apply(&vec_Y);
    std::cout << vec_interpol.adjoint() << std::endl;
    std::cout << vec_quad.adjoint() << std::endl;
    ASSERT_TRUE(vec_interpol.isApprox(vec_quad));
  }
}

}  // namespace spacetime
