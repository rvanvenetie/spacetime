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
using Time::ortho_tree;
using Time::OrthonormalWaveletFn;
using Time::three_point_tree;
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

int main(int argc, char* argv[]) {
  std::string problem;
  size_t initial_refines = 0;
  size_t max_dofs = 0;
  boost::program_options::options_description problem_optdesc(
      "Problem options");
  problem_optdesc.add_options()(
      "problem", po::value<std::string>(&problem)->default_value("smooth"))(
      "initial_refines", po::value<size_t>(&initial_refines))(
      "max_dofs", po::value<size_t>(&max_dofs)->default_value(
                      std::numeric_limits<std::size_t>::max()));

  AdaptiveHeatEquationOptions adapt_opts;
  boost::program_options::options_description adapt_optdesc(
      "AdaptiveHeatEquation options");
  adapt_optdesc.add_options()("use_cache",
                              po::value<bool>(&adapt_opts.use_cache_))(
      "solve_rtol", po::value<double>(&adapt_opts.solve_rtol_))(
      "solve_maxit", po::value<size_t>(&adapt_opts.solve_maxit_))(
      "estimate_saturation_layers",
      po::value<size_t>(&adapt_opts.estimate_saturation_layers_))(
      "estimate_mean_zero", po::value<bool>(&adapt_opts.estimate_mean_zero_))(
      "mark_theta", po::value<double>(&adapt_opts.mark_theta_))(
      "PX_alpha", po::value<double>(&adapt_opts.P_X_alpha_))(
      "PX_inv",
      po::value<HeatEquationOptions::SpaceInverse>(&adapt_opts.P_X_inv_))(
      "PY_inv",
      po::value<HeatEquationOptions::SpaceInverse>(&adapt_opts.P_Y_inv_))(
      "PX_mg_cycles", po::value<size_t>(&adapt_opts.P_X_mg_cycles_))(
      "PY_mg_cycles", po::value<size_t>(&adapt_opts.P_Y_mg_cycles_));
  boost::program_options::options_description cmdline_options;
  cmdline_options.add(problem_optdesc).add(adapt_optdesc);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(),
            vm);
  po::notify(vm);
  std::cout << adapt_opts << std::endl;
  auto T = space::InitialTriangulation::UnitSquare(initial_refines);

  T.hierarch_basis_tree.UniformRefine(1);
  ortho_tree.UniformRefine(1);
  three_point_tree.UniformRefine(1);

  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      three_point_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
  X_delta.SparseRefine(1);

  std::pair<std::unique_ptr<LinearFormBase<Time::OrthonormalWaveletFn>>,
            std::unique_ptr<LinearFormBase<Time::ThreePointWaveletFn>>>
      problem_data;
  if (problem == "smooth")
    problem_data = SmoothProblem();
  else if (problem == "singular")
    problem_data = SingularProblem();
  else {
    std::cout << "problem not recognized :-(" << std::endl;
    return 1;
  }
  AdaptiveHeatEquation heat_eq(std::move(X_delta),
                               std::move(problem_data.first),
                               std::move(problem_data.second), adapt_opts);

  size_t ndof = 0;
  while (ndof < max_dofs) {
    auto start = std::chrono::steady_clock::now();
    auto [solution, solve_data] =
        heat_eq.Solve(heat_eq.vec_Xd_out()->ToVectorContainer());
    ndof = heat_eq.vec_Xd_out()->Bfs().size();  // A slight overestimate.
    auto [residual, residual_norm] = heat_eq.Estimate();
    auto end = std::chrono::steady_clock::now();
    auto marked_nodes = heat_eq.Mark();
    heat_eq.Refine(marked_nodes);
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "XDelta-size: " << ndof << " residual-norm: " << residual_norm
              << " total-memory-kB: " << getmem()
              << " solve-estimate-time: " << elapsed_seconds.count()
              << " solve-PCG-steps: " << solve_data.second << std::endl;
  }

  return 0;
}
