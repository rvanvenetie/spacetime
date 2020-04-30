
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "datastructures/multi_tree_view.hpp"
#include "space/initial_triangulation.hpp"
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
constexpr int iters = 300;

int main() {
  std::cout << "Sizeof comparison" << std::endl;
  std::cout << "unsigned int : " << sizeof(unsigned int) << std::endl;
  std::cout << "int : " << sizeof(int) << std::endl;
  std::cout << "uint : " << sizeof(uint) << std::endl;
  std::cout << "size_t : " << sizeof(size_t) << std::endl;
  std::cout << "Vertex * : " << sizeof(Vertex *) << std::endl;
  std::cout << "std::vector<Vertex *> : " << sizeof(std::vector<Vertex *>)
            << std::endl;
  std::cout << "SmallVector<Vertex *,4> : " << sizeof(SmallVector<Vertex *, 4>)
            << std::endl;
  std::cout << "SmallVector<Vertex *,8> : " << sizeof(SmallVector<Vertex *, 8>)
            << std::endl;
  std::cout << "StaticVector<Vertex *,4> : "
            << sizeof(StaticVector<Vertex *, 4>) << std::endl;
  std::cout << "std::array<Vertex *, 4> : " << sizeof(std::array<Vertex *, 4>)
            << std::endl;
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
