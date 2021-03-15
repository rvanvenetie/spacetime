#include "double_tree_view.hpp"

#include <cmath>
#include <cstdlib>
#include <map>
#include <set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "time/bases.hpp"

using namespace space;
using namespace datastructures;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::Eq;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

constexpr int max_level = 5;

TEST(DoubleTreeView, project) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto elements = T.elem_meta_root->Bfs();
  auto vertices = T.vertex_meta_root->Bfs();

  auto db_tree =
      DoubleTreeView<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);
  db_tree.DeepRefine();

  // Project on the first axis.
  auto dt_proj_0 = db_tree.Project_0();
  auto dt_proj_0_nodes = dt_proj_0->Bfs();
  ASSERT_EQ(dt_proj_0_nodes.size(), vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i)
    ASSERT_EQ(dt_proj_0_nodes[i]->node(), vertices[i]);

  // Project on the second axis.
  auto dt_proj_1 = db_tree.Project_1();
  auto dt_proj_1_nodes = dt_proj_1->Bfs();
  ASSERT_EQ(dt_proj_1_nodes.size(), elements.size());
  for (size_t i = 0; i < elements.size(); ++i)
    ASSERT_EQ(dt_proj_1_nodes[i]->node(), elements[i]);
}

TEST(DoubleTreeView, Union) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto elements = T.elem_meta_root->Bfs();
  auto vertices = T.vertex_meta_root->Bfs();

  auto from_tree =
      DoubleTreeView<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);
  from_tree.DeepRefine();
  from_tree.ComputeFibers();
  auto to_tree =
      DoubleTreeView<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);

  // Copy axis 0 into `to_tree`.
  to_tree.Project_0()->Union(from_tree.Project_0());
  ASSERT_EQ(to_tree.Project_0()->Bfs().size(), vertices.size());

  // Copy all subtrees in axis 1 into `to_tree`.
  for (auto item : to_tree.Project_0()->Bfs(true)) {
    item->FrozenOtherAxis()->Union(from_tree.Fiber_1(item->node()));
  }
  ASSERT_EQ(to_tree.Bfs().size(), from_tree.Bfs().size());

  ASSERT_EQ(to_tree.root()->children(0)[0]->children(1)[0],
            to_tree.root()->children(1)[0]->children(0)[0]);
}

TEST(DoubleTreeVector, sum) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vec_sp =
      DoubleTreeVector<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);
  // Create a sparse DoubleTree Vector filled with random values.
  vec_sp.SparseRefine(4);
  for (const auto &nv : vec_sp.Bfs()) nv->set_random();

  // Create a double tree uniformly refined with levels [1,4].
  auto vec_unif =
      DoubleTreeVector<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);
  for (const auto &nv : vec_unif.Bfs()) nv->set_random();

  // Create two empty vectors holding the sum.
  auto vec_0 =
      DoubleTreeVector<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);
  auto vec_1 =
      DoubleTreeVector<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);

  // vec_0 = vec_sp + vec_unif
  vec_0 += vec_sp;
  vec_0 += vec_unif;

  // vec_1 = vec_unif + vec_sp
  vec_1 += vec_unif;
  vec_1 += vec_sp;

  // Assert vec_0 == vec_1
  auto vec_0_nodes = vec_0.Bfs();
  auto vec_1_nodes = vec_1.Bfs();
  ASSERT_EQ(vec_0_nodes.size(), vec_1_nodes.size());
  for (size_t i = 0; i < vec_0_nodes.size(); ++i)
    ASSERT_EQ(vec_0_nodes[i]->value(), vec_1_nodes[i]->value());

  // Assert that the sum domain is larger than the two vectors themselves.
  ASSERT_GE(vec_0_nodes.size(),
            std::max(vec_sp.Bfs().size(), vec_unif.Bfs().size()));

  // Calculate the sum by dict
  std::map<DoubleTreeVector<Vertex, Element2D>::Impl::TupleNodes, double>
      vec_dict;
  for (const auto &nv : vec_sp.Bfs()) vec_dict[nv->nodes()] = nv->value();
  for (const auto &nv : vec_unif.Bfs()) vec_dict[nv->nodes()] += nv->value();

  for (const auto &nv : vec_0_nodes)
    ASSERT_EQ(nv->value(), vec_dict[nv->nodes()]);
}

TEST(DoubleTreeVector, frozen_vector) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto elements = T.elem_meta_root->Bfs();
  auto vertices = T.vertex_meta_root->Bfs();

  auto db_tree =
      DoubleTreeVector<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);
  db_tree.DeepRefine();

  // Assert that main axis correspond to trees we've put in.
  auto dt_proj_0_nodes = db_tree.Project_0()->Bfs();
  auto dt_proj_1_nodes = db_tree.Project_1()->Bfs();

  ASSERT_EQ(dt_proj_0_nodes.size(), vertices.size());
  ASSERT_EQ(dt_proj_1_nodes.size(), elements.size());
  for (size_t i = 0; i < dt_proj_0_nodes.size(); ++i)
    ASSERT_EQ(dt_proj_0_nodes[i]->node(), vertices[i]);
  for (size_t i = 0; i < dt_proj_1_nodes.size(); ++i)
    ASSERT_EQ(dt_proj_1_nodes[i]->node(), elements[i]);

  // Initialize the vector ones.
  for (auto db_node : db_tree.Bfs()) db_node->set_value(1.0);
  for (auto db_node : db_tree.Bfs()) ASSERT_EQ(db_node->value(), 1.0);

  // Check that this also holds for the fibers.
  db_tree.ComputeFibers();
  for (auto labda : db_tree.Project_0()->Bfs()) {
    auto fiber = db_tree.Fiber_1(labda->node());
    for (auto f_node : fiber->Bfs()) ASSERT_EQ(f_node->value(), 1.0);
  }
  for (auto labda : db_tree.Project_1()->Bfs()) {
    auto fiber = db_tree.Fiber_0(labda->node());
    for (auto f_node : fiber->Bfs()) ASSERT_EQ(f_node->value(), 1.0);
  }

  // Check that the to_array is correct.
  auto dt_np = db_tree.ToVector();
  ASSERT_EQ(dt_np.size(), db_tree.Bfs().size());
  for (size_t i = 0; i < dt_np.size(); ++i) ASSERT_EQ(dt_np[i], 1.0);

  // Check that copying works.
  auto db_tree_copy = db_tree.DeepCopy();
  auto db_tree_copy_nodes = db_tree_copy.Bfs();
  for (auto db_node : db_tree_copy_nodes) ASSERT_EQ(db_node->value(), 1.0);
}

TEST(Gradedness, FullTensor) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.ortho_tree.UniformRefine(6);
  B.three_point_tree.UniformRefine(6);

  for (int lvl_t = 0; lvl_t < 3; lvl_t++) {
    for (int lvl_x = 0; lvl_x < 6; lvl_x++) {
      auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
          B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
      X_delta.UniformRefine({lvl_t, lvl_x});

      ASSERT_EQ(X_delta.Gradedness(), lvl_x + 1);
    }
  }
}

TEST(Gradedness, SparseTensor) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.ortho_tree.UniformRefine(6);
  B.three_point_tree.UniformRefine(6);

  int level = 8;
  for (int L = 1; L < 5; L++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    X_delta.SparseRefine(level, {L, 1});

    ASSERT_EQ(X_delta.Gradedness(), L);
  }
}

TEST(XDelta, SparseRefine) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  B.three_point_tree.UniformRefine(max_level);
  T.hierarch_basis_tree.UniformRefine(2 * max_level);
  std::vector<int> n_lvl_t;
  std::vector<int> n_lvl_x;
  for (auto nodes : B.three_point_tree.NodesPerLevel())
    n_lvl_t.push_back(nodes.size());
  for (auto nodes : T.hierarch_basis_tree.NodesPerLevel())
    n_lvl_x.push_back(nodes.size());

  for (int L = 1; L <= max_level; L++) {
    // Reset the underlying trees.
    auto B = Time::Bases();
    auto T = space::InitialTriangulation::UnitSquare();

    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    X_delta.SparseRefine(2 * L, {2, 1}, /* grow_tree */ true);

    auto ndofs = X_delta.Bfs().size();
    size_t ndofs_expected = 0;
    for (int L_t = 0; L_t <= 2 * L; L_t++)
      for (int L_x = 0; L_x <= 2 * L; L_x++)
        if (2 * L_t + L_x <= 2 * L)
          ndofs_expected += n_lvl_t.at(L_t) * n_lvl_x.at(L_x);

    ASSERT_EQ(ndofs, ndofs_expected);
  }
}

TEST(XDelta, UniformRefine) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  B.three_point_tree.UniformRefine(max_level);
  T.hierarch_basis_tree.UniformRefine(2 * max_level);
  std::vector<int> n_t{0};
  std::vector<int> n_x{0};
  for (auto nodes : B.three_point_tree.NodesPerLevel())
    n_t.push_back(nodes.size() + n_t.back());
  for (auto nodes : T.hierarch_basis_tree.NodesPerLevel())
    n_x.push_back(nodes.size() + n_x.back());

  for (int L = 1; L <= max_level; L++) {
    // Reset the underlying trees.
    auto B = Time::Bases();
    auto T = space::InitialTriangulation::UnitSquare();
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    X_delta.UniformRefine({L, 2 * L}, /* grow_tree */ true);

    ASSERT_EQ(X_delta.Bfs().size(), n_t.at(L + 1) * n_x.at(2 * L + 1));
  }
}
