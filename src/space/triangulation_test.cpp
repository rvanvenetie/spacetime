#include "triangulation.hpp"

#include <cmath>
#include <map>
#include <set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "initial_triangulation.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

namespace space {
using ::testing::ElementsAre;

TEST(Triangulation, Refine) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(1);

  auto elements = T.elem_meta_root->Bfs();
  auto vertices = T.vertex_meta_root->Bfs();
  ASSERT_EQ(vertices.size(), 5);
  EXPECT_THAT(vertices[4]->parents(), ElementsAre(vertices[0], vertices[1]));
  EXPECT_THAT(vertices[0]->children(), ElementsAre(vertices[4]));
  EXPECT_THAT(vertices[1]->children(), ElementsAre(vertices[4]));

  elements[2]->Refine();
  elements = T.elem_meta_root->Bfs();
  vertices = T.vertex_meta_root->Bfs();
  ASSERT_TRUE(vertices[5]->on_domain_boundary);
  EXPECT_THAT(vertices[5]->parents(), ElementsAre(vertices[4]));
  EXPECT_THAT(vertices[4]->children(), ElementsAre(vertices[5]));

  elements[4]->Refine();
  elements = T.elem_meta_root->Bfs();
  vertices = T.vertex_meta_root->Bfs();
  ASSERT_TRUE(vertices[6]->on_domain_boundary);
  EXPECT_THAT(vertices[6]->parents(), ElementsAre(vertices[4]));
  EXPECT_THAT(vertices[4]->children(), ElementsAre(vertices[5], vertices[6]));

  elements[6]->Refine();
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
      std::set<Element2D*>(element_roots.begin(), element_roots.end());
  for (int i = 0; i < 100; ++i) {
    auto elem = *leaves.begin();
    leaves.erase(leaves.begin());
    elem->Refine();
    for (auto child : elem->children()) leaves.emplace(child);
  }

  for (auto vertex : init_triang.vertex_meta_root->Bfs()) {
    ASSERT_EQ(vertex->on_domain_boundary, (vertex->x == 0 || vertex->x == 1 ||
                                           vertex->y == 0 || vertex->y == 1));
  }

  for (auto elem : init_triang.elem_meta_root->Bfs()) {
    size_t bdr_vtx = 0;
    for (auto vertex : elem->vertices())
      if (vertex->x == 0 || vertex->x == 1 || vertex->y == 0 || vertex->y == 1)
        bdr_vtx++;
    ASSERT_EQ(elem->TouchesDomainBoundary(), bdr_vtx > 1);
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

  std::set<Element2D*> leaves(elements.begin(), elements.end());
  for (int i = 0; i < 200; ++i) {
    auto elem = *leaves.begin();
    leaves.erase(leaves.begin());
    elem->Refine();
    for (auto child : elem->children()) leaves.emplace(child);
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

TEST(Triangulation, RefineHierarchicalBasisFn) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(1);
  auto hb = T.hierarch_basis_tree.Bfs();
  auto vertices = T.vertex_tree.Bfs();
  auto elem = T.elem_tree.Bfs();
  ASSERT_EQ(hb.size(), 5);
  ASSERT_EQ(vertices.size(), 5);
  ASSERT_EQ(elem.size(), 6);

  T.hierarch_basis_tree.UniformRefine(2);
  hb = T.hierarch_basis_tree.Bfs();
  vertices = T.vertex_tree.Bfs();
  elem = T.elem_tree.Bfs();
  ASSERT_EQ(hb.size(), 9);
  ASSERT_EQ(vertices.size(), 9);
  ASSERT_EQ(elem.size(), 14);

  size_t ml = 10;
  for (int i = 0; i < 15; ++i) {
    auto T = InitialTriangulation::UnitSquare();
    T.hierarch_basis_tree.DeepRefine([ml](auto node) {
      return node->is_metaroot() || (node->level() < ml && bsd_rnd() % 3 != 0);
    });
    auto hb = T.hierarch_basis_tree.Bfs();
    auto vertices = T.vertex_tree.Bfs();
    ASSERT_EQ(hb.size(), vertices.size());
    for (int i = 0; i < hb.size(); ++i) {
      ASSERT_EQ(vertices[i], hb[i]->vertex());
    }
  }
}

TEST(Triangulation, BarycentricCoordinates) {
  auto T = InitialTriangulation::UnitSquare();

  size_t ml = 10;
  for (int i = 0; i < 15; ++i) {
    auto T = InitialTriangulation::UnitSquare();
    T.hierarch_basis_tree.DeepRefine([ml](auto node) {
      return node->is_metaroot() || (node->level() < ml && bsd_rnd() % 3 != 0);
    });
    for (auto elem : T.elem_tree.Bfs()) {
      for (auto v : elem->vertices()) {
        auto bary = elem->BarycentricCoordinates(v->x, v->y);
        ASSERT_TRUE((bary.array() >= 0).all());
        ASSERT_TRUE((bary.array() >= 1).any());
        auto found_coords = elem->GlobalCoordinates(bary[1], bary[2]);
        ASSERT_FLOAT_EQ(found_coords.first, v->x);
        ASSERT_FLOAT_EQ(found_coords.second, v->y);
      }
    }
  }
}

TEST(Triangulation, InitialRefinement) {
  auto T_0 = InitialTriangulation::UnitSquare();
  auto T_2 = InitialTriangulation::UnitSquare();
  T_2.elem_tree.UniformRefine(2);
  auto T_4 = InitialTriangulation::UnitSquare();
  T_4.elem_tree.UniformRefine(4);

  auto T_init_2 = InitialTriangulation::UnitSquare(2);
  size_t T_2_leaf_count = 0;
  for (auto elem : T_2.elem_tree.Bfs())
    if (elem->is_leaf()) T_2_leaf_count++;
  ASSERT_EQ(T_init_2.elem_tree.Bfs().size(), T_2_leaf_count);

  T_init_2.elem_tree.UniformRefine(2);
  size_t T_4_leaf_count = 0;
  for (auto elem : T_4.elem_tree.Bfs())
    if (elem->is_leaf()) T_4_leaf_count++;
  size_t bla_count = 0;
  for (auto elem : T_init_2.elem_tree.Bfs())
    if (elem->is_leaf()) bla_count++;
  ASSERT_EQ(bla_count, T_4_leaf_count);
}

}  // namespace space
