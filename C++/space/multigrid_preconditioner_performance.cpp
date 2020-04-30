#include <cmath>
#include <map>
#include <random>
#include <set>

#include "initial_triangulation.hpp"
#include "operators.hpp"
#include "triangulation.hpp"
#include "triangulation_view.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

using namespace space;
using namespace datastructures;

Eigen::VectorXd RandomVector(const TriangulationView &triang,
                             bool dirichlet_boundary = true) {
  Eigen::VectorXd vec(triang.V);
  vec.setRandom();
  if (dirichlet_boundary)
    for (int v = 0; v < triang.V; v++)
      if (triang.OnBoundary(v)) vec[v] = 0.0;
  return vec;
}

constexpr int level = 15;
constexpr int iters = 10;
constexpr int apply_iters = 10;

int main() {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(::level);

  for (size_t i = 0; i < iters; ++i) {
    // Create a random subtree
    auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
    vertex_subtree.DeepRefine(
        /* call_filter */ [](auto &&vertex) {
          return vertex->level() <= 0 || (bsd_rnd() % 5 < 3);
        });
    std::cout << vertex_subtree.Bfs().size() << "/"
              << T.vertex_meta_root->Bfs().size() << std::endl;

    auto T_view = TriangulationView(vertex_subtree.Bfs());
    auto mg_op = MultigridPreconditioner<StiffnessOperator>(T_view);
    auto v = RandomVector(T_view);
    for (size_t k = 0; k < apply_iters; k++) {
      mg_op.Apply(v);
    }
  }
}
