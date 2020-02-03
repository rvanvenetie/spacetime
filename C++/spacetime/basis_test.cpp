#include "basis.hpp"
#include "../space/initial_triangulation.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::ortho_tree;
using Time::OrthonormalWaveletFn;
using Time::three_point_tree;
using Time::ThreePointWaveletFn;

namespace spacetime {
TEST(GenerateYDelta, FullTensor) {
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
    auto Y_delta_fulltensor =
        DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>(
            ortho_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
    Y_delta_fulltensor.UniformRefine(level);

    auto found_nodes = Y_delta.Bfs();
    auto fulltensor_nodes = Y_delta_fulltensor.Bfs();

    ASSERT_EQ(found_nodes.size(), fulltensor_nodes.size());
    for (int i = 0; i < found_nodes.size(); ++i) {
      ASSERT_EQ(found_nodes[i]->nodes(), fulltensor_nodes[i]->nodes());
    }
  }
}

TEST(GenerateYDelta, SparseTensor) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  ortho_tree.UniformRefine(6);
  three_point_tree.UniformRefine(6);

  for (int level = 0; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.SparseRefine(level);

    auto Y_delta = GenerateYDelta(X_delta);
    auto Y_delta_fulltensor =
        DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>(
            ortho_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
    Y_delta_fulltensor.SparseRefine(level);

    auto found_nodes = Y_delta.Bfs();
    auto fulltensor_nodes = Y_delta_fulltensor.Bfs();

    ASSERT_EQ(found_nodes.size(), fulltensor_nodes.size());
    for (int i = 0; i < found_nodes.size(); ++i) {
      ASSERT_EQ(found_nodes[i]->nodes(), fulltensor_nodes[i]->nodes());
    }
  }
}
};  // namespace spacetime
