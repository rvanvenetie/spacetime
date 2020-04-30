#include "space/multigrid_triangulation_view.hpp"

#include <cmath>
#include <map>
#include <set>

#include "datastructures/multi_tree_vector.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "space/triangulation.hpp"

using namespace space;
using namespace datastructures;

constexpr int max_level = 6;

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

TEST(MultigridTriangulationView, UniformRefine) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);

  for (int level = 0; level <= max_level; ++level) {
    auto vertex_view = TreeView<Vertex>(T.vertex_meta_root);
    vertex_view.UniformRefine(level);

    std::vector<Vertex *> vertices;
    for (auto vtx : vertex_view.Bfs()) vertices.emplace_back(vtx->node());

    // Now create the corresponding element tree
    MultigridTriangulationView triang_view(vertices);

    // Lets see if this actually gives some fruitful results!
    ASSERT_EQ(triang_view.elements().size(), pow(2, level + 2) - 2);
    for (const auto &elem : triang_view.elements()) {
      ASSERT_TRUE(elem.level() <= level);
    }
  }
}

TEST(MultigridTriangulationView, VertexSubTree) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);

  // Create a subtree with only vertices lying below the diagonal.
  auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
  vertex_subtree.DeepRefine(/* call_filter */ [](const auto &vertex) {
    return vertex->level() == 0 || (vertex->x + vertex->y <= 1.0);
  });
  ASSERT_TRUE(vertex_subtree.Bfs().size() < T.vertex_meta_root->Bfs().size());
  std::vector<Vertex *> vertices;
  for (auto vtx : vertex_subtree.Bfs()) vertices.emplace_back(vtx->node());
  std::set<Vertex *> vertices_subtree;
  for (auto vertex : vertices) vertices_subtree.insert(vertex);

  auto T_view = MultigridTriangulationView(std::move(vertices));
  // Check all nodes necessary for the elem subtree are
  // inside the vertices_subtree.
  for (const auto &elem : T_view.elements()) {
    for (auto &vtx : elem.node()->vertices()) {
      ASSERT_TRUE(vertices_subtree.count(vtx));
    }
  }

  // And the other way around.
  const auto &elements = T_view.elements();
  std::set<Element2D *> elements_subtree;
  for (const auto &nv : elements) {
    elements_subtree.insert(nv.node());
  }
  ASSERT_EQ(elements_subtree.size(), elements.size());
  // Check all nodes necessary for the elem subtree are
  // inside the elements_subtree.
  for (auto &vertex : vertices_subtree) {
    for (auto &elem : vertex->patch) {
      ASSERT_TRUE(elements_subtree.count(elem));
    }
  }
}

std::vector<std::vector<Element2D *>> Transform(
    const std::vector<std::vector<Element2DView *>> &patches) {
  std::vector<std::vector<Element2D *>> result;
  for (const auto &patch : patches) {
    result.emplace_back();
    for (const auto &elem : patch) result.back().emplace_back(elem->node());
  }
  // Sort the patches.
  for (auto &patch : result) std::sort(patch.begin(), patch.end());
  return result;
}

TEST(MultigridTriangulationView, CoarseToFine) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  for (size_t j = 0; j < 5; ++j) {
    auto tree_view = TreeView<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
    tree_view.DeepRefine(
        /* call_filter */ [](auto &&nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });

    std::vector<Vertex *> vertices;
    for (auto vtx : tree_view.Bfs())
      vertices.emplace_back(vtx->node()->vertex());
    auto coarse =
        MultigridTriangulationView(vertices,
                                   /* initialize_finest_level */ false);

    // test a single Refine and Coarsen.
    coarse.Refine();
    coarse.Coarsen();
    ASSERT_EQ(Transform(coarse.patches()),
              Transform(MultigridTriangulationView(
                            vertices,
                            /* initialize_finest_level */ false)
                            .patches()));

    // Now refine all.
    while (coarse.CanRefine()) coarse.Refine();
    auto fine = MultigridTriangulationView(vertices,
                                           /* initialize_finest_level */ true);

    ASSERT_EQ(Transform(fine.patches()), Transform(coarse.patches()));
  }
}

TEST(MultigridTriangulationView, FineToCoarse) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  for (size_t j = 0; j < 5; ++j) {
    auto tree_view = TreeView<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
    tree_view.DeepRefine(
        /* call_filter */ [](auto &&nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });

    std::vector<Vertex *> vertices;
    for (auto vtx : tree_view.Bfs())
      vertices.emplace_back(vtx->node()->vertex());
    auto fine = MultigridTriangulationView(vertices,
                                           /* initialize_finest_level */ true);
    // test a single Coarsen and Refine.
    fine.Coarsen();
    fine.Refine();

    auto fine_copy = MultigridTriangulationView(
        vertices, /* initialize_finest_level */ true);
    ASSERT_EQ(Transform(fine.patches()), Transform(fine_copy.patches()));

    // Coarsen all.
    while (fine.CanCoarsen()) fine.Coarsen();
    auto coarse = MultigridTriangulationView(
        vertices, /* initialize_finest_level */ false);
    ASSERT_EQ(Transform(fine.patches()), Transform(coarse.patches()));
  }
}
