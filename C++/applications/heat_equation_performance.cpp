
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "../space/initial_triangulation.hpp"
#include "../time/bilinear_form.hpp"
#include "heat_equation.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

using namespace applications;
using namespace spacetime;
using namespace space;
using namespace Time;
using namespace datastructures;

constexpr int level = 12;
constexpr int heateq_iters = 5;
constexpr int inner_iters = 10;
constexpr bool use_cache = true;

int main() {
  auto B = Time::Bases();
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(::level);
  B.ortho_tree.UniformRefine(::level);
  B.three_point_tree.UniformRefine(::level);

  for (size_t j = 0; j < ::heateq_iters; ++j) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    X_delta.SparseRefine(::level);

    HeatEquation heat_eq(X_delta);

    // Generate some random input.
    for (auto nv : heat_eq.vec_X()->Bfs()) {
      if (nv->node_1()->on_domain_boundary()) continue;
      nv->set_random();
    }

    auto v_in = heat_eq.vec_X()->ToVectorContainer();
    for (size_t k = 0; k < ::inner_iters; k++) {
      heat_eq.S()->Apply(v_in);
    }
  }
  return 0;
}
