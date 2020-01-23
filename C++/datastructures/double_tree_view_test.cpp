#include "double_tree_view.hpp"

#include <cmath>
#include <cstdlib>
#include <map>
#include <set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/triangulation.hpp"

using namespace space;
using namespace datastructures;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::Eq;

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
  auto to_tree =
      DoubleTreeView<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);

  // Copy axis 0 into `to_tree`.
  to_tree.Project_0()->Union(from_tree.Project_0());
  ASSERT_EQ(to_tree.Project_0()->Bfs().size(), vertices.size());

  // Copy all subtrees in axis 1 into `to_tree`.
  for (auto item : to_tree.Project_0()->Bfs(true)) {
    item->FrozenOtherAxis()->Union(from_tree.Fiber(item));
  }
  ASSERT_EQ(to_tree.Bfs().size(), from_tree.Bfs().size());

  ASSERT_EQ(to_tree.root->children(0)[0]->children(1)[0],
            to_tree.root->children(1)[0]->children(0)[0]);
}

TEST(DoubleTreeVector, sum) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vec_sp =
      DoubleTreeVector<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);
  // Create a sparse DoubleTree Vector filled with random values.
  vec_sp.SparseRefine(4);
  for (const auto &nv : vec_sp.Bfs())
    nv->set_value(std::rand() * 1.0 / RAND_MAX);

  // Create a double tree uniformly refined with levels [1,4].
  auto vec_unif =
      DoubleTreeVector<Vertex, Element2D>(T.vertex_meta_root, T.elem_meta_root);
  for (const auto &nv : vec_unif.Bfs())
    nv->set_value(std::rand() * 1.0 / RAND_MAX);

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
  for (auto labda : db_tree.Project_0()->Bfs()) {
    auto fiber = db_tree.Fiber(labda);
    for (auto f_node : fiber->Bfs()) ASSERT_EQ(f_node->value(), 1.0);
  }
  for (auto labda : db_tree.Project_1()->Bfs()) {
    auto fiber = db_tree.Fiber(labda);
    for (auto f_node : fiber->Bfs()) ASSERT_EQ(f_node->value(), 1.0);
  }

  // Check that the to_array is correct.
  auto dt_np = db_tree.ToArray();
  ASSERT_EQ(dt_np.size(), db_tree.Bfs().size());
  for (auto val : dt_np) ASSERT_EQ(val, 1.0);

  // Check that copying works.
  auto db_tree_copy = db_tree.DeepCopy();
  auto db_tree_copy_nodes = db_tree_copy.Bfs();
  for (auto db_node : db_tree_copy_nodes) ASSERT_EQ(db_node->value(), 1.0);
}
