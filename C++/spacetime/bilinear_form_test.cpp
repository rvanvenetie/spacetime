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
    auto X_delta = DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
        three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.UniformRefine(level);

    //    auto Y_delta = GenerateYDelta(X_delta);
    //    auto Y_delta_fulltensor =
    //        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>(
    //            ortho_tree.meta_root.get(),
    //            T.hierarch_basis_tree.meta_root.get());
    //    Y_delta_fulltensor.UniformRefine(level);

    auto bil_form =
        BilinearForm<Time::MassOperator, space::MassOperator, decltype(X_delta),
                     decltype(&X_delta)>(X_delta, &X_delta);
  }
}

};  // namespace spacetime
