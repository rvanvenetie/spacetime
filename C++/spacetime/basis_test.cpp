#include "basis.hpp"

#include <set>

#include "../space/initial_triangulation.hpp"
#include "../time/haar_basis.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::haar_tree;
using Time::HaarWaveletFn;
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

    auto Y_delta = GenerateYDelta<DoubleTreeView>(X_delta);
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

    auto Y_delta = GenerateYDelta<DoubleTreeView>(X_delta);
    auto Y_delta_fulltensor =
        DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>(
            ortho_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
    Y_delta_fulltensor.SparseRefine(level);

    auto found_nodes = Y_delta.Bfs();
    auto fulltensor_nodes = Y_delta_fulltensor.Bfs();

    ASSERT_EQ(found_nodes.size(), fulltensor_nodes.size());
    for (int i = 0; i < found_nodes.size(); ++i)
      ASSERT_EQ(found_nodes[i]->nodes(), fulltensor_nodes[i]->nodes());
  }
}

TEST(GenerateXDeltaUnderscore, EqualsSparseGrid) {
  size_t max_level = 6;
  auto T = space::InitialTriangulation::UnitSquare();
  auto X_delta = DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
      three_point_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
  auto X_delta_next =
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
          three_point_tree.meta_root.get(),
          T.hierarch_basis_tree.meta_root.get());
  for (size_t level = 0; level < max_level; level++) {
    X_delta.SparseRefine(2 * level, {2, 1});
    auto X_delta_underscore = GenerateXDeltaUnderscore(X_delta);
    X_delta_next.SparseRefine(2 * (level + 1), {2, 1});
    auto Xdu_vertices = X_delta_underscore.Bfs();
    auto Xdn_vertices = X_delta_next.Bfs();
    ASSERT_EQ(Xdu_vertices.size(), Xdn_vertices.size());
  }
}

TEST(GenerateSigma, SmallSigma) {
  haar_tree.UniformRefine(2);

  auto Lambda = DoubleTreeVector<HaarWaveletFn, HaarWaveletFn>(
      haar_tree.meta_root.get(), haar_tree.meta_root.get());
  Lambda.UniformRefine(2);
  auto Sigma = GenerateSigma(Lambda, Lambda);
  auto sigma_nodes = Sigma->Bfs();

  auto test_dbltree = DoubleTreeVector<HaarWaveletFn, HaarWaveletFn>(
      haar_tree.meta_root.get(), haar_tree.meta_root.get());
  test_dbltree.UniformRefine({2, 2});
  auto test_nodes = test_dbltree.Bfs();

  ASSERT_EQ(sigma_nodes.size(), test_nodes.size());
  for (int i = 0; i < sigma_nodes.size(); ++i)
    ASSERT_EQ(sigma_nodes[i]->nodes(), test_nodes[i]->nodes());
}

TEST(GenerateSigma, FullTensorSigma) {
  for (int level = 0; level < 6; level++) {
    ortho_tree.UniformRefine(level);
    three_point_tree.UniformRefine(level);
    auto T = space::InitialTriangulation::UnitSquare();
    T.hierarch_basis_tree.UniformRefine(level);

    auto Lambda_in =
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>(
            ortho_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
    Lambda_in.UniformRefine(level);
    auto Lambda_out =
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
            three_point_tree.meta_root.get(),
            T.hierarch_basis_tree.meta_root.get());
    Lambda_out.UniformRefine(level);

    auto Sigma = GenerateSigma(Lambda_in, Lambda_out);
    auto sigma_nodes = Sigma->Bfs();

    auto test_dbltree =
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>(
            ortho_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
    test_dbltree.UniformRefine({level, level});
    auto test_nodes = test_dbltree.Bfs();

    ASSERT_EQ(sigma_nodes.size(), test_nodes.size());
    for (int i = 0; i < sigma_nodes.size(); ++i)
      ASSERT_EQ(sigma_nodes[i]->nodes(), test_nodes[i]->nodes());
  }
}

TEST(GenerateTheta, FullTensorTheta) {
  for (int level = 0; level < 6; level++) {
    ortho_tree.UniformRefine(level);
    three_point_tree.UniformRefine(level);
    auto T = space::InitialTriangulation::UnitSquare();
    T.hierarch_basis_tree.UniformRefine(level);

    auto Lambda_in =
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>(
            ortho_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
    Lambda_in.UniformRefine(level);
    auto Lambda_out =
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
            three_point_tree.meta_root.get(),
            T.hierarch_basis_tree.meta_root.get());
    Lambda_out.UniformRefine(level);

    auto Theta = GenerateTheta(Lambda_in, Lambda_out);
    auto theta_nodes = Theta->Bfs();

    auto test_dbltree =
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
            three_point_tree.meta_root.get(),
            T.hierarch_basis_tree.meta_root.get());
    test_dbltree.UniformRefine(level);
    auto test_nodes = test_dbltree.Bfs();

    ASSERT_EQ(theta_nodes.size(), test_nodes.size());
    for (int i = 0; i < theta_nodes.size(); ++i)
      ASSERT_EQ(theta_nodes[i]->nodes(), test_nodes[i]->nodes());
  }
}
};  // namespace spacetime
