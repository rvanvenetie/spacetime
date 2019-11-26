
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "datastructures/multi_tree_view.hpp"
#include "space/triangulation.hpp"
#include "space/triangulation_view.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

using namespace space;
using namespace datastructures;

constexpr int level = 10;
constexpr int iters = 1000;

int main() {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(::level);

  for (size_t i = 0; i < iters; ++i) {
    // Create a random subtree
    auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
    vertex_subtree.DeepRefine(
        /* call_filter */ [](auto &&vertex) {
          return vertex->level() <= 0 || bsd_rnd() % 3 != 0;
        });

    auto T_view = TriangulationView(vertex_subtree);
  }
}
