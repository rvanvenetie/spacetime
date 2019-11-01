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

TEST(Triangulation, TripleTreeView) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_meta_root->UniformRefine(3);

  auto vertices = T.vertex_meta_root->Bfs(/*include_metaroot*/ true);
  auto elements = T.elem_meta_root->Bfs(/*include_metaroot*/ true);

  auto multi_root = TripleNodeView<Vertex, Element2D, Vertex>::CreateRoot(
      T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);
  multi_root->Refine();

  ASSERT_EQ(multi_root->children(0).size(),
            T.vertex_meta_root->children().size());
  ASSERT_EQ(multi_root->children(1).size(),
            T.elem_meta_root->children().size());

  // Test uniform refine
  for (int level = 0; level < 3; ++level) {
    multi_root->UniformRefine(level);
    auto multi_nodes = multi_root->Bfs(/*include_metaroot*/ true);

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

  // Test deep refine
  multi_root->DeepRefine();
  auto multi_nodes = multi_root->Bfs(/*include_metaroot*/ true);
  ASSERT_EQ(multi_nodes.size(),
            vertices.size() * elements.size() * vertices.size());

  // Test some member functions.
  for (const auto &multi_node : multi_nodes) {
    auto [n1, n2, n3] = multi_node->nodes();
    ASSERT_EQ(multi_node->level(), n1->level() + n2->level() + n3->level());
    ASSERT_EQ(multi_node->is_root(), multi_node == multi_root);
    ASSERT_EQ(multi_node->is_metaroot(),
              n1->is_metaroot() || n2->is_metaroot() || n3->is_metaroot());
    ASSERT_EQ(multi_node->is_leaf(),
              n1->is_leaf() && n2->is_leaf() && n3->is_leaf());
  }

  // Test Copy.
  auto multi_root_copy = multi_root->DeepCopy();
  auto multi_nodes_copy = multi_root_copy->Bfs(/*include_metaroot */ true);
  ASSERT_EQ(multi_nodes.size(), multi_nodes_copy.size());
  for (int i = 0; i < multi_nodes.size(); ++i) {
    ASSERT_EQ(multi_nodes[i]->nodes(), multi_nodes_copy[i]->nodes());
  }
}
