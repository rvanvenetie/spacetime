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

namespace applications {
std::istream& operator>>(std::istream& in,
                         HeatEquationOptions::SpaceInverse& inverse_type) {
  std::string token;
  in >> token;
  if (token == "DirectInverse" || token == "di")
    inverse_type = HeatEquationOptions::SpaceInverse::DirectInverse;
  else if (token == "Multigrid" || token == "mg")
    inverse_type = HeatEquationOptions::SpaceInverse::Multigrid;
  else
    in.setstate(std::ios_base::failbit);
  return in;
}
}  // namespace applications

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
  size_t num_threads = 1;
  bool sparse_refine = true;
  bool calculate_condition_numbers = false;
  bool print_time_apply = false;
  double solve_rtol = 1e-6;
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
      "num_threads", po::value<size_t>(&num_threads))(
      "print_time_apply", po::value<bool>(&print_time_apply))(
      "calculate_condition_numbers",
      po::value<bool>(&calculate_condition_numbers));

  AdaptiveHeatEquationOptions adapt_opts;
  boost::program_options::options_description adapt_optdesc(
      "AdaptiveHeatEquation options");
  adapt_optdesc.add_options()("use_cache",
                              po::value<bool>(&adapt_opts.use_cache))(
      "build_space_mats", po::value<bool>(&adapt_opts.build_space_mats))(
      "solve_rtol", po::value<double>(&solve_rtol))(
      "solve_maxit", po::value<size_t>(&adapt_opts.solve_maxit))(
      "estimate_saturation_layers",
      po::value<size_t>(&adapt_opts.estimate_saturation_layers))(
      "estimate_mean_zero", po::value<bool>(&adapt_opts.estimate_mean_zero))(
      "mark_theta", po::value<double>(&adapt_opts.mark_theta))(
      "PX_alpha", po::value<double>(&adapt_opts.PX_alpha))(
      "PX_inv",
      po::value<HeatEquationOptions::SpaceInverse>(&adapt_opts.PX_inv))(
      "PY_inv",
      po::value<HeatEquationOptions::SpaceInverse>(&adapt_opts.PY_inv))(
      "PXY_mg_build", po::value<bool>(&adapt_opts.PXY_mg_build))(
      "PX_mg_cycles", po::value<size_t>(&adapt_opts.PX_mg_cycles))(
      "PY_mg_cycles", po::value<size_t>(&adapt_opts.PY_mg_cycles));
  boost::program_options::options_description cmdline_options;
  cmdline_options.add(problem_optdesc).add(adapt_optdesc);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(),
            vm);
  po::notify(vm);
  std::cout << "Problem options:" << std::endl;
  std::cout << "\tProblem: " << problem << std::endl;
  std::cout << "\tDomain: " << domain
            << "; initial-refines: " << initial_refines << std::endl;
  std::cout << std::endl;
  std::cout << adapt_opts << "\tsolve-rtol: " << solve_rtol << std::endl << std::endl;

  assert(num_threads > 0 && num_threads <= omp_get_max_threads() &&
         num_threads <= MAX_NUMBER_THREADS);
  omp_set_num_threads(num_threads);

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
    else if (problem == "moving-peak")
      problem_data = MovingPeakProblem(vec_Xd);
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
      std::cout << "\n\tlmin-PY-A: " << lanczos_Y.min()
                << "\n\tlmax-PY-A: " << lanczos_Y.max()
                << "\n\tlmin-PX-S: " << lanczos_X.min()
                << "\n\tlmax-PX-S: " << lanczos_X.max()
                << "\n\tcond-time: " << duration_cond.count() << std::endl;
      continue;
    }

    // Solve - estimate.
    auto start = std::chrono::steady_clock::now();
    auto [solution, pcg_data] =
        heat_eq.Solve(solve_rtol, tools::linalg::StoppingCriterium::Relative);
    std::chrono::duration<double> duration_solve =
        std::chrono::steady_clock::now() - start;
    std::cout << "\n\tsolve-PCG-steps: " << pcg_data.iterations
              << "\n\tsolve-time: " << duration_solve.count()
              << "\n\tsolve-memory: " << getmem() << std::flush;

    if (print_time_apply) {
      auto heat_d_dd = heat_eq.heat_d_dd();
      std::cout << "\n\tA-time-per-apply: " << heat_d_dd->A()->TimePerApply()
                << "\n\tB-time-per-apply: " << heat_d_dd->B()->TimePerApply()
                << "\n\tBT-time-per-apply: " << heat_d_dd->BT()->TimePerApply()
                << "\n\tG-time-per-apply: " << heat_d_dd->G()->TimePerApply()
                << "\n\tP_Y-time-per-apply: "
                << heat_d_dd->P_Y()->TimePerApply()
                << "\n\tP_X-time-per-apply: "
                << heat_d_dd->P_X()->TimePerApply()
                << "\n\tS-time-per-apply: " << heat_d_dd->S()->TimePerApply()
                << "\n\ttotal-time-apply: " << heat_d_dd->TotalTimeApply()
                << "\n\ttotal-time-construct: "
                << heat_d_dd->TotalTimeConstruct() << std::flush;
    }

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
