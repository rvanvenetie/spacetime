#include <cmath>
#include <map>
#include <random>
#include <set>

#include "bases.hpp"
#include "bilinear_form.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

using namespace Time;
using namespace datastructures;

constexpr int level = 10;
constexpr int bilform_iters = 10;
constexpr int inner_iters = 150;

int main() {
  std::cout << sizeof(int) << std::endl;
  std::cout << sizeof(long) << std::endl;
  std::cout << sizeof(long long) << std::endl;
  bool print_mesh = false;
  Bases B;
  size_t iter = 0;
  while (true) {
    std::cout << "iter: " << ++iter;
    if (print_mesh) {
      std::cout << "\n\tmesh: [";
      for (auto nv : B.ortho_tree.Bfs()) std::cout << nv->center() << ",";
      std::cout << "]" << std::flush;
    }
    // Refine towards the corner.
    for (auto psi : B.ortho_tree.Bfs())
      if (!psi->is_full() && psi->index() <= std::pow(1.3, psi->level()))
        psi->Refine();
    for (auto psi : B.three_point_tree.Bfs())
      if (!psi->is_full() && psi->index() <= std::pow(1.3, psi->level()))
        psi->Refine();

    std::cout << "\n\tortho-tree-size: " << B.ortho_tree.Bfs().size()
              << std::endl;
    std::cout << "\n\tthreept-tree-size: " << B.three_point_tree.Bfs().size()
              << std::endl;
    std::cout << std::endl;
  }
  return 0;
}
