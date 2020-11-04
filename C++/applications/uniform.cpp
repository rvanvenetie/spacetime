#include <stdlib.h>

#include <boost/program_options.hpp>
#include <chrono>
#include <limits>

#include "../space/initial_triangulation.hpp"
#include "../time/basis.hpp"
#include "../tools/util.hpp"
#include "adaptive_heat_equation.hpp"
#include "problems.hpp"

using applications::AdaptiveHeatEquation;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

using namespace applications;
namespace po = boost::program_options;

space::InitialTriangulation InitialTriangulation(std::string domain,
                                                 size_t initial_refines) {
  if (domain == "square" || domain == "unit-square")
    return space::InitialTriangulation::UnitSquare(initial_refines);
  else if (domain == "lshape" || domain == "l-shape")
    return space::InitialTriangulation::LShape(initial_refines);
  else {
    std::cout << "domain not recognized :-(" << std::endl;
    exit(1);
  }
}

int main(int argc, char* argv[]) {
  std::string problem, domain;
  size_t initial_refines = 0;
  size_t max_level = 0;
  size_t max_dofs = 0;
  bool sparse_refine = true;
  bool calculate_condition_numbers = false;
  boost::program_options::options_description problem_optdesc(
      "Problem options");
  problem_optdesc.add_options()(
      "problem", po::value<std::string>(&problem)->default_value("singular"))(
      "domain", po::value<std::string>(&domain)->default_value("square"))(
      "initial_refines", po::value<size_t>(&initial_refines))(
      "max_level",
      po::value<size_t>(&max_level)
          ->default_value(std::numeric_limits<std::size_t>::max()))(
      "max_dofs", po::value<size_t>(&max_dofs)->default_value(
                      std::numeric_limits<std::size_t>::max()))(
      "sparse_refine", po::value<bool>(&sparse_refine))(
      "calculate_condition_numbers",
      po::value<bool>(&calculate_condition_numbers));

  boost::program_options::options_description cmdline_options;
  cmdline_options.add(problem_optdesc);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(),
            vm);
  po::notify(vm);
  std::cout << "Problem options:" << std::endl;
  std::cout << "\tProblem: " << problem << std::endl;
  std::cout << "\tDomain: " << domain
            << "; initial-refines: " << initial_refines << std::endl;
  std::cout << std::endl;

  auto T = InitialTriangulation(domain, initial_refines);
  auto B = Time::Bases();
  auto vec_Xd = std::make_shared<
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());

  for (int level = 1; level < max_level; level++) {
    std::pair<std::unique_ptr<LinearFormBase<Time::OrthonormalWaveletFn>>,
              std::unique_ptr<LinearFormBase<Time::ThreePointWaveletFn>>>
        problem_data;
    if (problem == "smooth")
      problem_data = SmoothProblem();
    else if (problem == "singular")
      problem_data = SingularProblem();
    else if (problem == "cylinder")
      problem_data = CylinderProblem();
    else {
      std::cout << "problem not recognized :-(" << std::endl;
      return 1;
    }
    B.ortho_tree.UniformRefine(level + 1);
    B.three_point_tree.UniformRefine(level + 1);
    T.hierarch_basis_tree.UniformRefine(2 * (level + 1));
    if (sparse_refine)
      vec_Xd->SparseRefine(2 * level, {2, 1});
    else
      vec_Xd->UniformRefine({level, 2 * level});
    size_t ndof_X = vec_Xd->Bfs().size();  // A slight overestimate.
    int max_node_time = 0, max_node_space = 0;
    for (auto node : vec_Xd->Bfs()) {
      max_node_time =
          std::max(max_node_time, std::get<0>(node->nodes())->level());
      max_node_space =
          std::max(max_node_space, std::get<1>(node->nodes())->level());
    }
    int max_space_tree_lvl = 0;
    for (auto node : T.hierarch_basis_tree.Bfs())
      max_space_tree_lvl = std::max(max_space_tree_lvl, node->level());
    std::cout << ndof_X << " " << max_node_time << " " << max_node_space << " "
              << max_space_tree_lvl << std::endl;
    if (ndof_X == 0) continue;
    if (ndof_X > max_dofs) break;
    AdaptiveHeatEquationOptions adapt_opts;
    adapt_opts.use_cache = false;
    AdaptiveHeatEquation heat_eq(vec_Xd, std::move(problem_data.first),
                                 std::move(problem_data.second), adapt_opts);
    size_t ndof_Y = heat_eq.vec_Ydd()->Bfs().size();  // A slight overestimate.
    std::cout << "level: " << level << "\n\tXDelta-size: " << ndof_X
              << "\n\tXDelta-Gradedness: " << vec_Xd->Gradedness()
              << "\n\tYDeltaDelta-size: " << ndof_Y
              << "\n\ttotal-memory-kB: " << getmem() << std::flush;

    if (calculate_condition_numbers) {
      auto start = std::chrono::steady_clock::now();
      std::chrono::duration<double> duration_cond =
          std::chrono::steady_clock::now() - start;

      // Set the initial vector to something valid.
      heat_eq.vec_Ydd()->Reset();
      for (auto nv : heat_eq.vec_Ydd()->Bfs())
        if (!nv->node_1()->on_domain_boundary()) nv->set_random();
      auto lanczos_Y = tools::linalg::Lanczos(
          *heat_eq.heat_d_dd()->A(), *heat_eq.heat_d_dd()->P_Y(),
          heat_eq.vec_Ydd()->ToVectorContainer());

      // Set the initial vector to something valid.
      heat_eq.vec_Xd()->Reset();
      for (auto nv : heat_eq.vec_Xd()->Bfs())
        if (!nv->node_1()->on_domain_boundary()) nv->set_random();
      auto lanczos_X = tools::linalg::Lanczos(
          *heat_eq.heat_d_dd()->S(), *heat_eq.heat_d_dd()->P_X(),
          heat_eq.vec_Xd()->ToVectorContainer());
      std::cout << "\n\tcond-PY-A: " << lanczos_Y.cond()
                << "\n\tcond-PX-S: " << lanczos_X.cond()
                << "\n\tcond-time: " << duration_cond.count() << std::endl;
      continue;
    }

    // Solve - estimate.
    auto start = std::chrono::steady_clock::now();
    auto [solution, pcg_data] = heat_eq.Solve();
    std::chrono::duration<double> duration_solve =
        std::chrono::steady_clock::now() - start;
    std::cout << "\n\tsolve-PCG-steps: " << pcg_data.iterations
              << "\n\tsolve-time: " << duration_solve.count()
              << "\n\tsolve-memory: " << getmem() << std::flush;

    start = std::chrono::steady_clock::now();
    auto [residual, global_errors] = heat_eq.Estimate(solution);
    auto [residual_norm, global_error] = global_errors;
    std::chrono::duration<double> duration_estimate =
        std::chrono::steady_clock::now() - start;

    std::cout << "\n\tresidual-norm: " << residual_norm
              << "\n\testimate-time: " << duration_estimate.count()
              << "\n\testimate-memory: " << getmem() << std::flush;
    std::cout << "\n\tglobal-error: " << global_error.error
              << "\n\tYnorm-error: " << global_error.error_Yprime
              << "\n\tT0-error: " << global_error.error_t0 << std::flush;

#ifdef VERBOSE
    std::cerr << std::endl << "Adaptive::Trees" << std::endl;
    std::cerr << "  T.vertex:   #bfs =  " << T.vertex_tree.Bfs().size()
              << std::endl;
    std::cerr << "  T.element:  #bfs =  " << T.elem_tree.Bfs().size()
              << std::endl;
    std::cerr << "  T.hierarch: #bfs =  " << T.hierarch_basis_tree.Bfs().size()
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "  B.elem:     #bfs =  " << B.elem_tree.Bfs().size()
              << std::endl;
    std::cerr << "  B.three_pt: #bfs =  " << B.three_point_tree.Bfs().size()
              << std::endl;
    std::cerr << "  B.ortho:    #bfs =  " << B.ortho_tree.Bfs().size()
              << std::endl;
#endif
    std::cout << std::endl;
  }

  return 0;
}
