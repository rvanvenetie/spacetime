#include "bilinear_form.hpp"

#include "../space/initial_triangulation.hpp"
#include "../space/operators.hpp"
#include "../time/linear_operator.hpp"
#include "basis.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::ortho_tree;
using Time::OrthonormalWaveletFn;
using Time::three_point_tree;
using Time::ThreePointWaveletFn;

namespace spacetime {
TEST(BilinearForm, XDeltaYDeltaFullTensorSparse) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  ortho_tree.UniformRefine(6);
  three_point_tree.UniformRefine(6);

  for (int level = 0; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.UniformRefine(level);
    auto Y_delta = GenerateYDelta(X_delta);

    auto vec_in = X_delta.template DeepCopy<
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>();
    auto vec_out = Y_delta.template DeepCopy<
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>();

    auto bil_form = CreateBilinearForm<Time::MassOperator, space::MassOperator>(
        vec_in, &vec_out);
    bil_form.Apply();
  }
}

};  // namespace spacetime
