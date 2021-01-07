#include <cmath>
#include <map>
#include <random>
#include <set>

#include "../tools/linalg.hpp"
#include "../tools/util.hpp"
#include "bilinear_form.hpp"
#include "initial_triangulation.hpp"
#include "linear_form.hpp"
#include "operators.hpp"
#include "triangulation.hpp"
#include "triangulation_view.hpp"

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

void UniformRefine(TreeVector<HierarchicalBasisFn> &tree_vec) {
  for (auto nv : tree_vec.Bfs()) {
    if (!nv->node()->is_full()) nv->node()->Refine();
    if (!nv->template is_full<0>()) nv->Refine();
  }
}

auto Mark(TreeVector<HierarchicalBasisFn> &residual, double theta = 0.8) {
  auto nodes = residual.Bfs();
  std::sort(nodes.begin(), nodes.end(), [](auto n1, auto n2) {
    return std::abs(n1->value()) > std::abs(n2->value());
  });
  double sq_norm = 0.0;
  for (auto node : nodes) sq_norm += node->value() * node->value();
  double cur_sq_norm = 0.0;
  size_t last_idx = 0;
  for (; last_idx < nodes.size(); last_idx++) {
    cur_sq_norm += nodes[last_idx]->value() * nodes[last_idx]->value();
    if (cur_sq_norm >= theta * theta * sq_norm) break;
  }
  nodes.resize(last_idx + 1);
  return nodes;
}

constexpr int level = 15;
constexpr int pcg_iters = 4;
constexpr int create_iters = 5;
constexpr bool print_mesh = false;

int main() {
  auto T = InitialTriangulation::LShape();
  auto f = [](double x, double y) { return 1; };

  auto x_d = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
  x_d.DeepRefine();
  auto solution = x_d.ToVector();

  size_t iter = 0;
  while (true) {
    std::cout << "iter: " << ++iter << "\n\tXDelta-size: " << x_d.Bfs().size()
              << "\n\ttotal-memory-kB: " << getmem() << std::flush;

    if (print_mesh) {
      std::cout << "\n\tmesh: [";
      for (auto nv : x_d.Bfs())
        std::cout << "[" << nv->node()->center().first << ","
                  << nv->node()->center().second << "],";
      std::cout << "]" << std::flush;
    }

    // Create linform + bilform + precond.
    auto lf = LinearForm(f, /* apply_quadrature*/ true, /*order*/ 2,
                         /* dirichlet_boundary */ true);
    space::OperatorOptions space_opts({.build_mat = false, .mg_cycles = 2});
    auto bilform = CreateBilinearForm<StiffnessOperator>(x_d, x_d, space_opts);
    auto precond =
        CreateBilinearForm<MultigridPreconditioner<StiffnessOperator>>(
            x_d, x_d, space_opts);

    // Calculate rhs.
    lf.Apply(x_d.root());
    auto rhs = x_d.ToVector();

    // Solve.
    auto [new_solution, pcg_data] =
        tools::linalg::PCG(bilform, rhs, precond, solution, pcg_iters, 1e-16);

    std::cout << "\n\tsolve-PCG-steps: " << pcg_data.iterations
              << "\n\tsolve-PCG-relative-residual: "
              << pcg_data.relative_residual
              << "\n\tsolve-PCG-algebraic-error: " << pcg_data.algebraic_error
              << std::flush;

    // Estimate.
    auto x_dd = x_d.DeepCopy();
    x_dd.FromVector(new_solution);
    UniformRefine(x_dd);
    std::cout << "\n\tXDeltaDelta-size: " << x_dd.Bfs().size();
    bilform = CreateBilinearForm<StiffnessOperator>(x_dd, x_dd, space_opts);
    bilform.Apply();
    auto residual = x_dd.ToVector();
    lf.Apply(x_dd.root());
    residual -= x_dd.ToVector();
    std::cout << "\n\tresidual-norm: " << residual.norm() << std::flush;

    // Mark.
    x_dd.FromVector(residual);
    auto marked_nodes = Mark(x_dd);
    std::cout << "\n\tmarked-nodes: " << marked_nodes.size() << std::flush;

    // Refine.
    x_d.FromVector(new_solution);
    x_d.ConformingRefinement(x_dd, marked_nodes);
    solution = x_d.ToVector();
    std::cout << std::endl;
  }
  return 0;
}
