
#include "space/triangulation_view.hpp"

#include <cmath>
#include <map>
#include <set>

#include "datastructures/multi_tree_view.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "space/triangulation.hpp"

using namespace space;
using namespace datastructures;
using ::testing::ElementsAre;

constexpr int max_level = 6;

TEST(TriangulationView, UniformRefine) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);

  for (int level = 0; level <= max_level; ++level) {
    auto vertex_view = TreeView<Vertex>(T.vertex_meta_root);
    vertex_view.UniformRefine(level);

    std::vector<Vertex *> vertices;
    for (auto vtx : vertex_view.Bfs()) vertices.emplace_back(vtx->node());

    // Now create the corresponding element tree
    TriangulationView triang_view(std::move(vertices));

    // Lets see if this actually gives some fruitful results!
    ASSERT_EQ(triang_view.element_leaves().size(), pow(2, level + 1));

    for (const auto &[elem, _] : triang_view.element_leaves()) {
      ASSERT_TRUE(elem->level() == level);
    }
  }
}

TEST(TriangulationView, VertexSubTree) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);

  // Create a subtree with only vertices lying below the diagonal.
  auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
  vertex_subtree.DeepRefine(/* call_filter */ [](const auto &vertex) {
    return vertex->level() == 0 || (vertex->x + vertex->y <= 1.0);
  });
  ASSERT_TRUE(vertex_subtree.Bfs().size() < T.vertex_meta_root->Bfs().size());

  auto T_view = TriangulationView(vertex_subtree);

  // Check there are no duplicates.
  std::set<Vertex *> vertices_subtree;
  for (auto vertex : T_view.vertices()) {
    vertices_subtree.insert(vertex);
  }
  ASSERT_EQ(vertices_subtree.size(), T_view.vertices().size());
  // Check all nodes necessary for the elem subtree are
  // inside the vertices_subtree.
  for (const auto &[elem, _] : T_view.element_leaves()) {
    for (auto &vtx : elem->vertices()) {
      ASSERT_TRUE(vertices_subtree.count(vtx));
    }
  }
}
