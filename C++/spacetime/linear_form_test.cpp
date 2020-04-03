#include "linear_form.hpp"

#include "../tools/integration.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "time/basis.hpp"
#include "time/orthonormal_basis.hpp"

constexpr int max_level = 6;

namespace spacetime {
TEST(LinearForm, Quadrature) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  Time::ortho_tree.UniformRefine(max_level);

  std::function<double(double)> time_f([](double t) { return t * t * t; });
  std::function<double(double, double)> space_f(
      [](double x, double y) { return x * y; });

  auto linform = CreateQuadratureLinearForm<Time::OrthonormalWaveletFn, 3, 2>(
      time_f, space_f);
  for (int level = 1; level < max_level; level++) {
    auto vec = datastructures::DoubleTreeVector<Time::OrthonormalWaveletFn,
                                                space::HierarchicalBasisFn>(
        Time::ortho_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    vec.SparseRefine(level);
    linform.Apply(&vec);
    for (auto phi : vec.Bfs())
      if (!std::get<1>(phi->nodes())->vertex()->on_domain_boundary)
        ASSERT_NE(phi->value(), 0.0);
  }
}

TEST(LinearForm, ZeroEval) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  Time::ortho_tree.UniformRefine(max_level);

  std::function<double(double, double)> space_f(
      [](double x, double y) { return x * y; });

  auto linform =
      CreateZeroEvalLinearForm<Time::OrthonormalWaveletFn, 2>(space_f);
  for (int level = 1; level < max_level; level++) {
    auto vec = datastructures::DoubleTreeVector<Time::OrthonormalWaveletFn,
                                                space::HierarchicalBasisFn>(
        Time::ortho_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    vec.SparseRefine(level);
    linform.Apply(&vec);
    ASSERT_NE(vec.ToVector().sum(), 0.0);
  }
}
}  // namespace spacetime
