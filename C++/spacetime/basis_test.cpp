#include "basis.hpp"

#include <set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "time/bases.hpp"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::HaarWaveletFn;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

namespace spacetime {
TEST(GenerateYDelta, FullTensor) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.ortho_tree.UniformRefine(6);
  B.three_point_tree.UniformRefine(6);

  for (int level = 0; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    X_delta.UniformRefine(level);

    auto Y_delta = GenerateYDelta<DoubleTreeView>(X_delta);
    auto Y_delta_fulltensor =
        DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>(
            B.ortho_tree.meta_root(), T.hierarch_basis_tree.meta_root());
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
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.ortho_tree.UniformRefine(6);
  B.three_point_tree.UniformRefine(6);

  for (int level = 0; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    X_delta.SparseRefine(level);

    auto Y_delta = GenerateYDelta<DoubleTreeView>(X_delta);
    auto Y_delta_fulltensor =
        DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>(
            B.ortho_tree.meta_root(), T.hierarch_basis_tree.meta_root());
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
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  auto X_delta = DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  auto X_delta_next =
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
          B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
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
  auto B = Time::Bases();
  B.haar_tree.UniformRefine(2);

  auto Lambda = DoubleTreeVector<HaarWaveletFn, HaarWaveletFn>(
      B.haar_tree.meta_root(), B.haar_tree.meta_root());
  Lambda.UniformRefine(2);
  auto Sigma = GenerateSigma(Lambda, Lambda);
  auto sigma_nodes = Sigma->Bfs();

  auto test_dbltree = DoubleTreeVector<HaarWaveletFn, HaarWaveletFn>(
      B.haar_tree.meta_root(), B.haar_tree.meta_root());
  test_dbltree.UniformRefine({2, 2});
  auto test_nodes = test_dbltree.Bfs();

  ASSERT_EQ(sigma_nodes.size(), test_nodes.size());
  for (int i = 0; i < sigma_nodes.size(); ++i)
    ASSERT_EQ(sigma_nodes[i]->nodes(), test_nodes[i]->nodes());
}

TEST(GenerateSigma, FullTensorSigma) {
  auto B = Time::Bases();
  for (int level = 0; level < 6; level++) {
    B.ortho_tree.UniformRefine(level);
    B.three_point_tree.UniformRefine(level);
    auto T = space::InitialTriangulation::UnitSquare();
    T.hierarch_basis_tree.UniformRefine(level);

    auto Lambda_in =
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>(
            B.ortho_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    Lambda_in.UniformRefine(level);
    auto Lambda_out =
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
            B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    Lambda_out.UniformRefine(level);

    auto Sigma = GenerateSigma(Lambda_in, Lambda_out);
    auto sigma_nodes = Sigma->Bfs();

    auto test_dbltree =
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>(
            B.ortho_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    test_dbltree.UniformRefine({level, level});
    auto test_nodes = test_dbltree.Bfs();

    ASSERT_EQ(sigma_nodes.size(), test_nodes.size());
    for (int i = 0; i < sigma_nodes.size(); ++i)
      ASSERT_EQ(sigma_nodes[i]->nodes(), test_nodes[i]->nodes());
  }
}

TEST(GenerateTheta, FullTensorTheta) {
  auto B = Time::Bases();
  for (int level = 0; level < 6; level++) {
    B.ortho_tree.UniformRefine(level);
    B.three_point_tree.UniformRefine(level);
    auto T = space::InitialTriangulation::UnitSquare();
    T.hierarch_basis_tree.UniformRefine(level);

    auto Lambda_in =
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>(
            B.ortho_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    Lambda_in.UniformRefine(level);
    auto Lambda_out =
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
            B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    Lambda_out.UniformRefine(level);

    auto Theta = GenerateTheta(Lambda_in, Lambda_out);
    auto theta_nodes = Theta->Bfs();

    auto test_dbltree =
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>(
            B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    test_dbltree.UniformRefine(level);
    auto test_nodes = test_dbltree.Bfs();

    ASSERT_EQ(theta_nodes.size(), test_nodes.size());
    for (int i = 0; i < theta_nodes.size(); ++i)
      ASSERT_EQ(theta_nodes[i]->nodes(), test_nodes[i]->nodes());
  }
}
};  // namespace spacetime
