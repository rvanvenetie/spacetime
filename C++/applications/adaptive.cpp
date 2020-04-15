#include <chrono>

#include "../space/initial_triangulation.hpp"
#include "../time/basis.hpp"
#include "../tools/util.hpp"
#include "adaptive_heat_equation.hpp"
#include "problems.hpp"

using applications::AdaptiveHeatEquation;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::ortho_tree;
using Time::OrthonormalWaveletFn;
using Time::three_point_tree;
using Time::ThreePointWaveletFn;

using namespace applications;

int main() {
  auto T = space::InitialTriangulation::UnitSquare();
  auto [g_lf, u0_lf] = SmoothProblem();

  T.hierarch_basis_tree.UniformRefine(1);
  ortho_tree.UniformRefine(1);
  three_point_tree.UniformRefine(1);

  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      three_point_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
  X_delta.SparseRefine(1);

  AdaptiveHeatEquation heat_eq(std::move(X_delta), std::move(g_lf),
                               std::move(u0_lf));

  while (true) {
    auto start = std::chrono::steady_clock::now();
    auto solution = heat_eq.Solve(heat_eq.vec_Xd_out()->ToVectorContainer());
    auto ndof = solution->container().size();  // A O(sqrt(N)) overestimate.
    auto [residual, residual_norm] = heat_eq.Estimate(/*mean_zero*/ false);
    auto end = std::chrono::steady_clock::now();
    auto marked_nodes = heat_eq.Mark();
    heat_eq.Refine(marked_nodes);
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "XDelta-size: " << ndof << " residual-norm: " << residual_norm
              << " total-memory-kB: " << getmem()
              << " solve-estimate-time: " << elapsed_seconds.count()
              << std::endl;
  }

  return 0;
}
