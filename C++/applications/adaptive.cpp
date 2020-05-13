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

int main(int argc, char* argv[]) {
  std::string problem;
  size_t initial_refines = 0;
  size_t max_dofs = 0;
  bool estimate_global_error = true;
  boost::program_options::options_description problem_optdesc(
      "Problem options");
  problem_optdesc.add_options()(
      "problem", po::value<std::string>(&problem)->default_value("singular"))(
      "initial_refines", po::value<size_t>(&initial_refines))(
      "max_dofs", po::value<size_t>(&max_dofs)->default_value(
                      std::numeric_limits<std::size_t>::max()))(
      "estimate_global_error", po::value<bool>(&estimate_global_error));

  AdaptiveHeatEquationOptions adapt_opts;
  boost::program_options::options_description adapt_optdesc(
      "AdaptiveHeatEquation options");
  adapt_optdesc.add_options()("use_cache",
                              po::value<bool>(&adapt_opts.use_cache))(
      "build_space_mats", po::value<bool>(&adapt_opts.build_space_mats))(
      "solve_rtol", po::value<double>(&adapt_opts.solve_rtol))(
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
  std::cout << adapt_opts << std::endl;
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare(initial_refines);

  T.hierarch_basis_tree.UniformRefine(1);
  B.ortho_tree.UniformRefine(1);
  B.three_point_tree.UniformRefine(1);

  auto vec_Xd = std::make_shared<
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  vec_Xd->SparseRefine(1);

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
  AdaptiveHeatEquation heat_eq(vec_Xd, std::move(problem_data.first),
                               std::move(problem_data.second), adapt_opts);

  size_t ndof = 0;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(vec_Xd->container().size());
  while (ndof < max_dofs) {
    ndof = vec_Xd->Bfs().size();  // A slight overestimate.
    std::cout << "XDelta-size: " << ndof << " total-memory-kB: " << getmem()
              << std::flush;

    // Solve - estimate.
    auto start = std::chrono::steady_clock::now();
    auto [solution, pcg_data] = heat_eq.Solve(x0);
    std::chrono::duration<double> duration_solve =
        std::chrono::steady_clock::now() - start;
    std::cout << " solve-PCG-steps: " << pcg_data.iterations
              << " solve-time: " << duration_solve.count()
              << " solve-memory: " << getmem() << std::flush;

    if (estimate_global_error) {
      start = std::chrono::steady_clock::now();
      auto [global_error, terms] = heat_eq.EstimateGlobalError(solution);
      std::chrono::duration<double> duration_global =
          std::chrono::steady_clock::now() - start;
      std::cout << " global-error: " << global_error
      << " Ynorm-error: " << terms.first
      << " T0-error: " << terms.second
                << " global-time: " << duration_global.count() << std::flush;
    }
    start = std::chrono::steady_clock::now();
    auto [residual, residual_norm] = heat_eq.Estimate(solution);
    std::chrono::duration<double> duration_estimate =
        std::chrono::steady_clock::now() - start;

    std::cout << " residual-norm: " << residual_norm
              << " estimate-time: " << duration_estimate.count()
              << " estimate-memory: " << getmem() << std::flush;

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

    // Mark - Refine.
    auto marked_nodes = heat_eq.Mark(residual);
    vec_Xd->FromVectorContainer(solution);

    start = std::chrono::steady_clock::now();
    heat_eq.Refine(marked_nodes);
    std::chrono::duration<double> duration_refine =
        std::chrono::steady_clock::now() - start;
    x0 = vec_Xd->ToVectorContainer();

    std::cout << " refine-time: " << duration_refine.count() << std::endl;
  }

  return 0;
}
