#include "multi_tree_view.hpp"

#include <cmath>
#include <map>
#include <set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/triangulation.hpp"

using namespace space;
using namespace datastructures;
using ::testing::ElementsAre;

constexpr int max_level = 4;

TEST(MultiNodeView, SingleRefine) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vertices = T.vertex_meta_root->Bfs(/*include_metaroot*/ true);
  auto elements = T.elem_meta_root->Bfs(/*include_metaroot*/ true);

  // Test single refine
  auto multi_tree = TripleTreeView<Vertex, Element2D, Vertex>(
      T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);
  auto multi_root = multi_tree.root;

  multi_root->Refine();
  ASSERT_EQ(multi_root->children(0).size(),
            T.vertex_meta_root->children().size());
  ASSERT_EQ(multi_root->children(1).size(),
            T.elem_meta_root->children().size());
}

TEST(MultiNodeView, UniformRefine) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vertices = T.vertex_meta_root->Bfs(/*include_metaroot*/ true);
  auto elements = T.elem_meta_root->Bfs(/*include_metaroot*/ true);

  // Test uniform refine
  for (int level = 0; level <= max_level; ++level) {
    auto multi_tree = TripleTreeView<Vertex, Element2D, Vertex>(
        T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);

    multi_tree.UniformRefine(level);
    auto multi_nodes = multi_tree.Bfs(/*include_metaroot*/ true);

    // Check that the levels of all multi nodes are indeed <= level.
    ASSERT_TRUE(
        std::all_of(multi_nodes.begin(), multi_nodes.end(), [&](auto nv) {
          auto levels = nv->levels();
          return (levels[0] <= level && levels[1] <= level &&
                  levels[2] <= level);
        }));

    // Count the underlying tree.
    auto N_vert_level =
        std::count_if(vertices.begin(), vertices.end(),
                      [&](auto n) { return n->level() <= level; });
    auto N_elem_level =
        std::count_if(elements.begin(), elements.end(),
                      [&](auto n) { return n->level() <= level; });
    ASSERT_EQ(multi_nodes.size(), N_vert_level * N_elem_level * N_vert_level);
  }

  // Test uniform refine in a single axis
  for (int i = 0; i < 3; ++i) {
    for (int level = 0; level <= max_level; ++level) {
      auto multi_tree = TripleTreeView<Vertex, Element2D, Vertex>(
          T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);

      std::array<int, 3> levels;
      for (int i = 0; i < 3; ++i) levels[i] = -1;
      levels[i] = level;
      multi_tree.UniformRefine(levels);
      auto multi_nodes = multi_tree.Bfs(/*include_metaroot*/ true);

      // Check that the levels of all multi nodes are indeed <= level.
      ASSERT_TRUE(
          std::all_of(multi_nodes.begin(), multi_nodes.end(), [&](auto nv) {
            auto levels = nv->levels();
            return (levels[i] <= level);
          }));

      // Count the underlying tree.
      if (i == 0 || i == 2) {
        auto N_vert_level =
            std::count_if(vertices.begin(), vertices.end(),
                          [&](auto n) { return n->level() <= level; });
        ASSERT_EQ(multi_nodes.size(), N_vert_level);
      } else {
        auto N_elem_level =
            std::count_if(elements.begin(), elements.end(),
                          [&](auto n) { return n->level() <= level; });
        ASSERT_EQ(multi_nodes.size(), N_elem_level);
      }
    }
  }
}

TEST(MultiNodeView, SparseRefine) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vertices = T.vertex_meta_root->Bfs(/*include_metaroot*/ true);
  auto elements = T.elem_meta_root->Bfs(/*include_metaroot*/ true);

  // Test sparse refine.
  std::map<int, int> elements_on_level, vertices_on_level;
  for (const auto &vertex : vertices) vertices_on_level[vertex->level()] += 1;
  for (const auto &elem : elements) elements_on_level[elem->level()] += 1;
  for (int level = 0; level <= max_level; ++level) {
    auto multi_tree = TripleTreeView<Vertex, Element2D, Vertex>(
        T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);

    std::array<int, 3> weights = {1, 1, 2};
    multi_tree.SparseRefine(level, weights);

    // Calculate the number of elements we expect.
    int count = 0;
    for (int i = -1; i <= max_level; ++i)
      for (int j = -1; j <= max_level; ++j)
        for (int k = -1; k <= max_level; ++k)
          if (weights[0] * i + weights[1] * j + weights[2] * k <= level) {
            count += vertices_on_level[i] * elements_on_level[j] *
                     vertices_on_level[k];
          }

    auto multi_nodes = multi_tree.Bfs(/*include_metaroot*/ true);
    ASSERT_EQ(multi_nodes.size(), count);
  }
}

TEST(MultiNodeView, DeepRefine) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vertices = T.vertex_meta_root->Bfs(/*include_metaroot*/ true);
  auto elements = T.elem_meta_root->Bfs(/*include_metaroot*/ true);

  // Test deep refine
  auto multi_tree = TripleTreeView<Vertex, Element2D, Vertex>(
      T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);
  multi_tree.DeepRefine();
  auto multi_nodes = multi_tree.Bfs(/*include_metaroot*/ true);
  ASSERT_EQ(multi_nodes.size(),
            vertices.size() * elements.size() * vertices.size());

  // Test some member functions.
  for (const auto &multi_node : multi_nodes) {
    auto [n1, n2, n3] = multi_node->nodes();
    ASSERT_EQ(multi_node->level(), n1->level() + n2->level() + n3->level());
    ASSERT_EQ(multi_node->is_root(), multi_node == multi_tree.root.get());
    ASSERT_EQ(multi_node->is_metaroot(),
              n1->is_metaroot() || n2->is_metaroot() || n3->is_metaroot());
    ASSERT_EQ(multi_node->is_leaf(),
              n1->is_leaf() && n2->is_leaf() && n3->is_leaf());
  }
}

TEST(MultiNodeView, DeepCopy) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vertices = T.vertex_meta_root->Bfs(/*include_metaroot*/ true);
  auto elements = T.elem_meta_root->Bfs(/*include_metaroot*/ true);

  // DeepRefine
  auto multi_tree = TripleTreeView<Vertex, Element2D, Vertex>(
      T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);
  multi_tree.DeepRefine();
  auto multi_nodes = multi_tree.Bfs(/*include_metaroot*/ true);

  // Test Copy.
  auto multi_root_copy = multi_tree.DeepCopy();
  auto multi_nodes_copy = multi_root_copy.Bfs(/*include_metaroot */ true);
  ASSERT_EQ(multi_nodes.size(), multi_nodes_copy.size());
  for (int i = 0; i < multi_nodes.size(); ++i) {
    ASSERT_EQ(multi_nodes[i]->nodes(), multi_nodes_copy[i]->nodes());
  }
}

TEST(MultiNodeView, Union) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vertices = T.vertex_meta_root->Bfs(/*include_metaroot*/ true);
  auto elements = T.elem_meta_root->Bfs(/*include_metaroot*/ true);

  // Test union
  auto union_tree = TripleTreeView<Vertex, Element2D, Vertex>(
      T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);

  // Union a lot of sparse trees.
  for (int i = -1; i <= max_level; ++i)
    for (int j = -1; j <= max_level; ++j)
      for (int k = -1; k <= max_level; ++k)
        if (i + j + k <= max_level) {
          auto multi_tree = TripleTreeView<Vertex, Element2D, Vertex>(
              T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);
          multi_tree.UniformRefine({i, j, k});
          union_tree.root->Union(multi_tree.root);
        }

  // Get the union nodes.
  auto union_nodes = union_tree.Bfs(/*include_metaroot*/ true);

  // Do the same, but now not stupid.
  auto multi_tree = TripleTreeView<Vertex, Element2D, Vertex>(
      T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);
  multi_tree.SparseRefine(max_level);

  // Compare the results
  auto multi_nodes = multi_tree.Bfs(/*include_metaroot*/ true);
  ASSERT_EQ(multi_nodes.size(), union_nodes.size());
  for (int i = 0; i < multi_nodes.size(); ++i) {
    ASSERT_EQ(multi_nodes[i]->nodes(), union_nodes[i]->nodes());
  }
}
