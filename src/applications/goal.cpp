
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

std::pair<Eigen::VectorXd, Eigen::VectorXd> Solve(AdaptiveHeatEquation& heat_eq,
                                                  Eigen::VectorXd init_sol,
                                                  double& t_delta) {
  Eigen::VectorXd solution = init_sol;
  double residual_norm = 0;
  int cycle = 1;

  auto start = std::chrono::steady_clock::now();
  auto rhs = heat_eq.RHS();
  std::chrono::duration<double> duration_rhs =
      std::chrono::steady_clock::now() - start;
  std::cout << "\n\trhs-time: " << duration_rhs.count();
  std::cout << "\n\trhs-g-linform-time: "
            << heat_eq.g_lin_form()->TimeLastApply();
  std::cout << "\n\trhs-u0-linform-time: "
            << heat_eq.u0_lin_form()->TimeLastApply();
  do {
    t_delta /= heat_eq.opts().solve_factor;
    std::cout << "\n\tcycle: " << cycle << "\n\t\tt_delta: " << t_delta;
    // Solve.
    start = std::chrono::steady_clock::now();
    auto [cur_solution, pcg_data] = heat_eq.Solve(solution, rhs, t_delta);
    solution = cur_solution;
    t_delta = pcg_data.algebraic_error;
    std::chrono::duration<double> duration_solve =
        std::chrono::steady_clock::now() - start;
    std::cout << "\n\t\tsolve-PCG-steps: " << pcg_data.iterations
              << "\n\t\tsolve-PCG-initial-algebraic-error: "
              << pcg_data.initial_algebraic_error
              << "\n\t\tsolve-PCG-algebraic-error: " << pcg_data.algebraic_error
              << "\n\t\tsolve-time: " << duration_solve.count()
              << "\n\t\tsolve-memory: " << getmem() << std::flush;

    // Estimate.
    start = std::chrono::steady_clock::now();
    auto [residual, global_errors] = heat_eq.Estimate(solution);
    residual_norm = global_errors.first;
    std::chrono::duration<double> duration_estimate =
        std::chrono::steady_clock::now() - start;

    std::cout << "\n\t\tresidual-norm: " << residual_norm
              << "\n\t\testimate-time: " << duration_estimate.count()
              << "\n\t\testimate-memory: " << getmem() << std::flush;
    cycle++;
  } while (t_delta > heat_eq.opts().solve_xi * (residual_norm + t_delta));
  t_delta = residual_norm + t_delta;

  return {solution, heat_eq.vec_Xdd()->ToVectorContainer()};
}

int main(int argc, char* argv[]) {
  AdaptiveHeatEquationOptions adapt_opts;
  boost::program_options::options_description adapt_optdesc(
      "AdaptiveHeatEquation options");
  adapt_optdesc.add_options()("use_cache",
                              po::value<bool>(&adapt_opts.use_cache))(
      "build_space_mats", po::value<bool>(&adapt_opts.build_space_mats))(
      "solve_factor", po::value<double>(&adapt_opts.solve_factor))(
      "solve_xi", po::value<double>(&adapt_opts.solve_xi))(
      "solve_maxit", po::value<size_t>(&adapt_opts.solve_maxit))(
      "mark_theta", po::value<double>(&adapt_opts.mark_theta))(
      "PX_mg_cycles", po::value<size_t>(&adapt_opts.PX_mg_cycles))(
      "PY_mg_cycles", po::value<size_t>(&adapt_opts.PY_mg_cycles));
  boost::program_options::options_description cmdline_options;
  cmdline_options.add(adapt_optdesc);
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(),
            vm);
  po::notify(vm);
  std::cout << adapt_opts << std::endl;

  auto T = space::InitialTriangulation::UnitSquare();
  auto B = Time::Bases();

  T.hierarch_basis_tree.UniformRefine(1);
  B.ortho_tree.UniformRefine(1);
  B.three_point_tree.UniformRefine(1);

  auto vec_Xd = std::make_shared<
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  vec_Xd->SparseRefine(1);

  auto [g_lin_form, u0_lin_form] = SmoothProblem();

  auto J_func = [](double t, double x, double y) {
    if (x < 0.1 || y < 0.1)
      return 1.;
    else
      return 0.;
  };

  AdaptiveHeatEquation heat_eq_u(vec_Xd, std::move(g_lin_form),
                                 std::move(u0_lin_form), adapt_opts);

  size_t ndof_Xd = 0;
  Eigen::VectorXd sol_u = Eigen::VectorXd::Zero(vec_Xd->container().size());
  Eigen::VectorXd sol_p = Eigen::VectorXd::Zero(vec_Xd->container().size());
  double t_delta_u = 1;
  double t_delta_p = 1;
  std::cout << "t_init u: " << t_delta_u << std::endl;
  std::cout << "t_init p: " << t_delta_p << std::endl;

  size_t iter = 0;
  auto start_algorithm = std::chrono::steady_clock::now();
  while (true) {
    // A slight overestimate.
    ndof_Xd = vec_Xd->Bfs().size();
    size_t ndof_Xdd = heat_eq_u.vec_Xdd()->Bfs().size();
    size_t ndof_Ydd = heat_eq_u.vec_Ydd()->Bfs().size();
    std::cout << "iter: " << ++iter << "\n\tXDelta-size: " << ndof_Xd
              << "\n\tXDeltaDelta-size: " << ndof_Xdd
              << "\n\tYDeltaDelta-size: " << ndof_Ydd
              << "\n\ttotal-memory-kB: " << getmem() << std::flush;

    Eigen::VectorXd residual_u, residual_p;

    // Solve for u.
    std::cout << "\n  solve-u:";
    std::tie(sol_u, residual_u) = Solve(heat_eq_u, sol_u, t_delta_u);

    // Create tmp heat equation object for p and solve.
    auto heat_eq_p = std::make_unique<AdaptiveHeatEquation>(
        vec_Xd, heat_eq_u.vec_Xdd(), heat_eq_u.vec_Ydd(), heat_eq_u.heat_d_dd(),
        std::make_unique<NoOpLinearForm<Time::OrthonormalWaveletFn>>(),
        std::make_unique<
            spacetime::InterpolationLinearForm<Time::ThreePointWaveletFn>>(
            vec_Xd, J_func),
        adapt_opts);
    std::cout << "\n  solve-p:";
    std::tie(sol_p, residual_p) = Solve(*heat_eq_p, sol_p, t_delta_p);
    vec_Xd->FromVectorContainer(sol_u);
    std::cout << "\n\tgoal:"
              << (heat_eq_p->u0_lin_form()->Apply(vec_Xd.get()).dot(sol_u))
              << std::endl;
    heat_eq_p.reset();

    // Combine the residual.
    Eigen::VectorXd residual_comb(residual_u.size());
    for (int i = 0; i < residual_u.size(); ++i)
      residual_comb[i] =
          sqrt(residual_u[i] * residual_u[i] + residual_p[i] * residual_p[i]);

    // Mark
    heat_eq_u.vec_Xdd()->FromVectorContainer(residual_comb);
    auto marked_nodes = heat_eq_u.Mark(heat_eq_u.vec_Xdd().get());

    auto start = std::chrono::steady_clock::now();
    // Refine. First, copy solution of p.
    vec_Xd->FromVectorContainer(sol_p);
    auto vec_Xd_p = vec_Xd->DeepCopy();

    // Then, refine and prolongate u.
    vec_Xd->FromVectorContainer(sol_u);
    auto r_info = heat_eq_u.Refine(marked_nodes);
    sol_u = vec_Xd->ToVectorContainer();

    // Finally, prolongate also p.
    vec_Xd->Reset();
    *vec_Xd += vec_Xd_p;
    sol_p = vec_Xd->ToVectorContainer();

    // Now, update
    std::chrono::duration<double> duration_refine =
        std::chrono::steady_clock::now() - start;

    std::cout << "\n\tnodes-marked: " << r_info.nodes_marked
              << "\n\tnodes-conforming: " << r_info.nodes_conforming
              << "\n\trefine-time: " << duration_refine.count()
              << "\n\ttotal-time-algorithm: "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start_algorithm)
                     .count()
              << std::endl;
  }

  return 0;
}
