#include <boost/program_options.hpp>
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
namespace po = boost::program_options;

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
    if (cur_sq_norm >= theta * theta * sq_norm) {
      nodes.resize(last_idx + 1);
      break;
    }
  }
  return nodes;
}

auto Now() { return std::chrono::steady_clock::now(); }
double Duration(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(Now() - start).count();
}

constexpr size_t mg_cycles = 4;
constexpr size_t mg_inner_cycles = 5;

int main(int argc, char *argv[]) {
  double mark_theta = 0.5;
  bool print_mesh = false;
  bool print_mesh_info = true;
  bool time_mg_cycles = true;
  int max_iter = 4;
  boost::program_options::options_description adapt_optdesc("Adaptive options");
  adapt_optdesc.add_options()("mark_theta", po::value<double>(&mark_theta))(
      "print_mesh", po::value<bool>(&print_mesh))("max_iters",
                                                  po::value<int>(&max_iter))(
      "print_mesh_info", po::value<bool>(&print_mesh_info))(
      "time_mg_cycles", po::value<bool>(&time_mg_cycles));
  boost::program_options::options_description cmdline_options;
  cmdline_options.add(adapt_optdesc);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(),
            vm);
  po::notify(vm);
  std::cout << "Adaptive options:" << std::endl;
  std::cout << "\ttheta: " << mark_theta << std::endl;
  std::cout << "\tmax_iter: " << max_iter << std::endl;

  auto T = InitialTriangulation::LShape();
  auto f = [](double x, double y) { return 1; };

  auto x_d = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
  x_d.DeepRefine();
  auto solution = x_d.ToVector();

  auto lf = LinearForm(f, /* apply_quadrature*/ true, /*order*/ 0,
                       /* dirichlet_boundary */ true);
  space::OperatorOptions space_opts(
      {.build_mat = false, .mg_cycles = mg_cycles});
  size_t iter = 0;
  auto start_algorithm = std::chrono::steady_clock::now();
  while (true) {
    auto start_iteration = std::chrono::steady_clock::now();
    std::cout << "iter: " << ++iter << "\n\tXDelta-size: " << x_d.Bfs().size()
              << "\n\ttotal-memory-kB: " << getmem() << std::flush;

    if (print_mesh_info) {
      auto time_start = std::chrono::steady_clock::now();
      TriangulationView t_view(x_d.Bfs());
      std::vector<int> hist;
      for (auto &&[elem, _] : t_view.element_leaves()) {
        if (elem->level() >= hist.size()) hist.resize(elem->level() + 1, 0);
        hist[elem->level()]++;
      }
      std::cout << "\n\ttime-tview-XDelta: " << Duration(time_start);
      std::cout << "\n\tmesh-info: [";
      for (auto val : hist) std::cout << val << ",";
      std::cout << "]" << std::flush;
    }

    if (print_mesh) {
      std::cout << "\n\tmesh: [";
      for (auto nv : x_d.Bfs())
        std::cout << "[" << nv->node()->center().first << ","
                  << nv->node()->center().second << "],";
      std::cout << "]" << std::flush;
    }

    // Create linform + bilform + precond.
    auto bilform = CreateBilinearForm<StiffnessOperator>(x_d, x_d, space_opts);
    auto precond =
        CreateBilinearForm<MultigridPreconditioner<StiffnessOperator>>(
            x_d, x_d, space_opts);
    std::cout << "\n\ttime-stiff-create: " << bilform.TimeCreate()
              << "\n\ttime-MG(" << mg_cycles
              << ")-create: " << precond.TimeCreate();

    // Calculate rhs.
    auto time_start = std::chrono::steady_clock::now();
    lf.Apply(x_d.root());
    auto rhs = x_d.ToVector();
    std::cout << "\n\ttime-rhs: " << Duration(time_start) << std::flush;

    // Solve.
    time_start = std::chrono::steady_clock::now();
    auto [new_solution, pcg_data] =
        tools::linalg::PCG(bilform, rhs, precond, solution, max_iter, 1e-16);
    std::cout << "\n\ttime-solve: " << Duration(time_start)
              << "\n\tsolve-PCG-steps: " << pcg_data.iterations
              << "\n\tsolve-PCG-relative-residual: "
              << pcg_data.relative_residual
              << "\n\tsolve-PCG-algebraic-error: " << pcg_data.algebraic_error
              << "\n\tsolve-memory: " << getmem()
              << "\n\ttime-stiff-per-apply: " << bilform.TimePerApply()
              << "\n\ttime-MG(" << mg_cycles
              << ")-per-apply: " << precond.TimePerApply();

    // Time preconds apply time.
    if (time_mg_cycles) {
      time_start = std::chrono::steady_clock::now();
      for (size_t mg = 1; mg < mg_cycles; ++mg) {
        auto mg_bilform =
            CreateBilinearForm<MultigridPreconditioner<StiffnessOperator>>(
                x_d, x_d, {.build_mat = false, .mg_cycles = mg});
        for (int i = 0; i < mg_inner_cycles; i++) mg_bilform.Apply();
        std::cout << "\n\ttime-MG(" << mg
                  << ")-create: " << mg_bilform.TimeCreate() << std::flush;
        std::cout << "\n\ttime-MG(" << mg
                  << ")-per-apply: " << mg_bilform.TimePerApply() << std::flush;
      }
      std::cout << "\n\ttime-mg-test: " << Duration(time_start);
    }

    // Estimate.
    time_start = std::chrono::steady_clock::now();
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
    std::cout << "\n\ttime-estimate: " << Duration(time_start)
              << "\n\testimate-memory: " << getmem() << std::flush;

    // Mark.
    time_start = std::chrono::steady_clock::now();
    x_dd.FromVector(residual);
    auto marked_nodes = Mark(x_dd, mark_theta);
    std::cout << "\n\tmarked-nodes: " << marked_nodes.size()
              << "\n\ttime-mark: " << Duration(time_start) << std::flush;

    // Refine.
    time_start = std::chrono::steady_clock::now();
    x_d.FromVector(new_solution);
    x_d.ConformingRefinement(x_dd, marked_nodes);
    solution = x_d.ToVector();
    std::cout << "\n\ttime-refine: " << Duration(time_start)
              << "\n\ttime-iteration: " << Duration(start_iteration)
              << "\n\ttime-total-algorithm: " << Duration(start_algorithm)
              << std::endl;
  }
  return 0;
}
