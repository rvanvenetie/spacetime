#include "triangulation.hpp"

#include <cmath>
#include <map>
#include <set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace space {
using ::testing::ElementsAre;

TEST(Triangulation, Refine) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(1);

  auto elements = T.elem_meta_root->Bfs();
  auto vertices = T.vertex_meta_root->Bfs();
  EXPECT_THAT(vertices[4]->parents(), ElementsAre(vertices[0], vertices[1]));
  EXPECT_THAT(vertices[0]->children(), ElementsAre(vertices[4]));
  EXPECT_THAT(vertices[1]->children(), ElementsAre(vertices[4]));

  T.elem_tree.Refine(elements[2]);
  elements = T.elem_meta_root->Bfs();
  vertices = T.vertex_meta_root->Bfs();
  ASSERT_TRUE(vertices[5]->on_domain_boundary);
  EXPECT_THAT(vertices[5]->parents(), ElementsAre(vertices[4]));
  EXPECT_THAT(vertices[4]->children(), ElementsAre(vertices[5]));

  T.elem_tree.Refine(elements[4]);
  elements = T.elem_meta_root->Bfs();
  vertices = T.vertex_meta_root->Bfs();
  ASSERT_TRUE(vertices[6]->on_domain_boundary);
  EXPECT_THAT(vertices[6]->parents(), ElementsAre(vertices[4]));
  EXPECT_THAT(vertices[4]->children(), ElementsAre(vertices[5], vertices[6]));

  T.elem_tree.Refine(elements[6]);
  elements = T.elem_meta_root->Bfs();
  vertices = T.vertex_meta_root->Bfs();
  ASSERT_FALSE(vertices[8]->on_domain_boundary);
  EXPECT_THAT(vertices[8]->parents(), ElementsAre(vertices[5], vertices[7]));
}

TEST(Triangulation, Area) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(2);

  auto elements = T.elem_meta_root->Bfs();
  auto vertices = T.vertex_meta_root->Bfs();
  ASSERT_EQ(elements[0]->area(), 0.5);
  ASSERT_EQ(elements[1]->area(), 0.5);

  ASSERT_EQ(elements[2]->area(), 0.25);
  ASSERT_EQ(elements[3]->area(), 0.25);
  ASSERT_EQ(elements[4]->area(), 0.25);
  ASSERT_EQ(elements[5]->area(), 0.25);

  ASSERT_EQ(elements[6]->area(), 0.125);
}

TEST(Triangulation, OnDomainBoundary) {
  auto init_triang = InitialTriangulation::UnitSquare();
  for (auto vertex : init_triang.vertex_meta_root->Bfs()) {
    ASSERT_TRUE(vertex->on_domain_boundary);
  }

  auto element_roots = init_triang.elem_meta_root->Bfs();
  auto leaves =
      std::set<Element2DPtr>(element_roots.begin(), element_roots.end());
  for (int i = 0; i < 100; ++i) {
    auto elem = *leaves.begin();
    leaves.erase(leaves.begin());
    auto children = init_triang.elem_tree.Refine(elem);
    leaves.insert(children.begin(), children.end());
  }

  for (auto vertex : init_triang.vertex_meta_root->Bfs()) {
    ASSERT_EQ(vertex->on_domain_boundary, (vertex->x == 0 || vertex->x == 1 ||
                                           vertex->y == 0 || vertex->y == 1));
  }
}

TEST(Triangulation, UniformRefinement) {
  auto T = InitialTriangulation::UnitSquare();
  auto elem_meta_root = T.elem_meta_root;
  ASSERT_TRUE(elem_meta_root->is_full());
  for (auto root : elem_meta_root->children()) {
    ASSERT_EQ(root->level(), 0);
  }

  T.elem_tree.UniformRefine(5);
  ASSERT_EQ(elem_meta_root->Bfs().size(),
            (std::pow(2, 6) - 1) * elem_meta_root->children().size());

  std::map<int, int> counts;
  for (auto elem : elem_meta_root->Bfs()) {
    ASSERT_LE(elem->level(), 5);
    counts[elem->level()] += 1;
  }

  for (int level = 0; level < 6; ++level) {
    ASSERT_EQ(counts[level], pow(2, level) * elem_meta_root->children().size());
  }
}

TEST(Triangulation, VertexPatch) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(1);

  auto elements = T.elem_meta_root->Bfs();
  auto vertices = T.vertex_meta_root->Bfs();
  ASSERT_EQ(elements.size(), 6);
  ASSERT_EQ(vertices.size(), 5);
  EXPECT_THAT(vertices[0]->patch, ElementsAre(elements[0]));
  EXPECT_THAT(vertices[1]->patch, ElementsAre(elements[1]));
  EXPECT_THAT(vertices[2]->patch, ElementsAre(elements[0], elements[1]));
  EXPECT_THAT(vertices[3]->patch, ElementsAre(elements[0], elements[1]));
  EXPECT_THAT(vertices[4]->patch,
              ElementsAre(elements[2], elements[3], elements[4], elements[5]));

  std::set<Element2DPtr> leaves(elements.begin(), elements.end());
  for (int i = 0; i < 200; ++i) {
    auto elem = *leaves.begin();
    leaves.erase(leaves.begin());
    auto children = T.elem_tree.Refine(elem);
    leaves.insert(children.begin(), children.end());
  }

  vertices = T.vertex_meta_root->Bfs();
  for (auto v : vertices) {
    if (v->level() == 0) continue;

    if (v->on_domain_boundary)
      ASSERT_EQ(v->patch.size(), 2);
    else
      ASSERT_EQ(v->patch.size(), 4);
  }
}
}  // namespace space
