#include "triangulation_view.hpp"

#include <cmath>
#include <map>
#include <set>

#include "datastructures/include.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "initial_triangulation.hpp"
#include "triangulation.hpp"

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

    // Now create the corresponding element tree
    TriangulationView triang_view(vertex_view);

    // Lets see if this actually gives some fruitful results!
    auto elements = triang_view.element_view().Bfs();
    ASSERT_EQ(elements.size(), pow(2, level + 2) - 2);

    for (auto elem : elements) {
      ASSERT_TRUE(elem->level() <= level);
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
  for (auto &elem : T_view.element_view().Bfs()) {
    for (auto &vtx : elem->node()->vertices()) {
      ASSERT_TRUE(vertices_subtree.count(vtx));
    }
  }

  // And the other way around.
  auto elements_view = T_view.element_view().Bfs();
  std::set<Element2D *> elements_subtree;
  for (auto &nv : elements_view) {
    elements_subtree.insert(nv->node());
  }
  ASSERT_EQ(elements_subtree.size(), elements_view.size());
  // Check all nodes necessary for the elem subtree are
  // inside the elements_subtree.
  for (auto &vertex : T_view.vertices()) {
    for (auto elem : vertex->patch) {
      ASSERT_TRUE(elements_subtree.count(elem));
    }
  }
}
