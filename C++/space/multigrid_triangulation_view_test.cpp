#include "space/multigrid_triangulation_view.hpp"

#include <cmath>
#include <map>
#include <set>

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

TEST(MultigridTriangulationView, CoarseToFine) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  for (size_t j = 0; j < 5; ++j) {
    auto tree_view = TreeView<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
    tree_view.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });

    auto triang = TriangulationView(tree_view.Bfs());
    auto coarse = MultigridTriangulationView::FromCoarsestTriangulation(triang);

    // test a single Refine and Coarsen.
    coarse.Refine();
    coarse.Coarsen();
    assert(coarse.patches() ==
           MultigridTriangulationView::FromCoarsestTriangulation(triang)
               .patches());

    // Now refine all.
    while (coarse.CanRefine()) coarse.Refine();
    auto fine = MultigridTriangulationView::FromFinestTriangulation(triang);
    assert(fine.patches() == coarse.patches());
  }
}

TEST(MultigridTriangulationView, FineToCoarse) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  for (size_t j = 0; j < 5; ++j) {
    auto tree_view = TreeView<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
    tree_view.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });

    auto triang = TriangulationView(tree_view.Bfs());
    auto fine = MultigridTriangulationView::FromFinestTriangulation(triang);
    // test a single Coarsen and Refine.
    fine.Coarsen();
    fine.Refine();
    assert(
        fine.patches() ==
        MultigridTriangulationView::FromFinestTriangulation(triang).patches());

    // Coarsen all.
    while (fine.CanCoarsen()) fine.Coarsen();
    auto coarse = MultigridTriangulationView::FromCoarsestTriangulation(triang);
    assert(fine.patches() == coarse.patches());
  }
}
