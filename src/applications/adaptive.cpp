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

void PrintTimeSliceSS(double t, AdaptiveHeatEquation::TypeXVector* solution) {
  auto time_slice = spacetime::Trace(t, *solution);

  // Calculate the triangulation corresponding to this space mesh.
  space::TriangulationView triang(time_slice.Bfs());
  std::cerr << "triang{";
  for (auto [elem, vertices] : triang.element_leaves())
    std::cerr << "(" << vertices[0] << ", " << vertices[1] << ", "
              << vertices[2] << ");";
  std::cerr << "}\t";

  // Calculate the single scale representation
  space::MassOperator op(triang);
  Eigen::VectorXd u_SS = time_slice.ToVector();
  assert(op.FeasibleVector(u_SS));
  op.ApplyHierarchToSingle(u_SS);
  assert(op.FeasibleVector(u_SS));

  // Print the data in single scale.
  time_slice.FromVector(u_SS);
  std::cerr << "vertices{";
  for (auto nv : time_slice.Bfs())
    std::cerr << "(" << nv->node()->center().first << ","
              << nv->node()->center().second << ") : " << nv->value() << ";";
  std::cerr << "}";
}

// Compile time constants.
constexpr size_t N_t = 20;
constexpr size_t N_x = 197;
constexpr size_t N_y = 199;

std::vector<std::tuple<float, float, float, double>> PrintSampling(
    AdaptiveHeatEquation::TypeXVector* solution) {
  int cnt[N_t + 1][N_x + 1][N_y + 1] = {0};
  double h_t = 1.0 / N_t;
  double h_x = 1.0 / N_x;
  double h_y = 1.0 / N_y;
  for (auto dblnode : solution->Bfs())
    for (int t = 0; t <= N_t; t++) {
      if (dblnode->node_0()->Eval(t * h_t) == 0) continue;
      for (int x = 0; x <= N_x; x++) {
        if (!dblnode->node_1()->Contains(x * h_x,
                                         dblnode->node_1()->center().second))
          continue;
        for (int y = 0; y <= N_y; y++) {
          if (dblnode->node_1()->Eval(x * h_x, y * h_y) == 0) continue;
          cnt[t][x][y]++;
        }
      }
    }

  std::vector<std::tuple<float, float, float, double>> result;
  for (int t = 0; t <= N_t; t++)
    for (int x = 0; x <= N_x; x++)
      for (int y = 0; y <= N_y; y++) {
        result.emplace_back(t * h_t, x * h_x, y * h_y, cnt[t][x][y]);
      }

  return result;
}

space::InitialTriangulation InitialTriangulation(std::string domain,
                                                 size_t initial_refines) {
  if (domain == "square" || domain == "unit-square")
    return space::InitialTriangulation::UnitSquare(initial_refines);
  else if (domain == "lshape" || domain == "l-shape")
    return space::InitialTriangulation::LShape(initial_refines);
  else if (domain == "pacman")
    return space::InitialTriangulation::Pacman(initial_refines);
  else {
    std::cout << "domain not recognized :-(" << std::endl;
    exit(1);
  }
}
}  // namespace applications

int main(int argc, char* argv[]) {
  std::string problem, domain;
  size_t initial_refines = 0;
  size_t max_dofs = 0;
  bool calculate_condition_numbers = false;
  bool print_centers = false;
  bool print_sampling = false;
  bool print_time_apply = false;
  std::vector<double> print_time_slices;
  boost::program_options::options_description problem_optdesc(
      "Problem options");
  problem_optdesc.add_options()(
      "problem", po::value<std::string>(&problem)->default_value("singular"))(
      "domain", po::value<std::string>(&domain)->default_value("square"))(
      "initial_refines", po::value<size_t>(&initial_refines))(
      "max_dofs", po::value<size_t>(&max_dofs)->default_value(
                      std::numeric_limits<std::size_t>::max()))(
      "calculate_condition_numbers",
      po::value<bool>(&calculate_condition_numbers))(
      "print_centers", po::value<bool>(&print_centers))(
      "print_sampling", po::value<bool>(&print_sampling))(
      "print_time_slices",
      po::value<std::vector<double>>(&print_time_slices)->multitoken())(
      "print_time_apply", po::value<bool>(&print_time_apply));

  std::sort(print_time_slices.begin(), print_time_slices.end());

  AdaptiveHeatEquationOptions adapt_opts;
  boost::program_options::options_description adapt_optdesc(
      "AdaptiveHeatEquation options");
  adapt_optdesc.add_options()("use_cache",
                              po::value<bool>(&adapt_opts.use_cache))(
      "build_space_mats", po::value<bool>(&adapt_opts.build_space_mats))(
      "solve_factor", po::value<double>(&adapt_opts.solve_factor))(
      "solve_xi", po::value<double>(&adapt_opts.solve_xi))(
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
  std::cout << adapt_opts << std::endl;

  auto T = InitialTriangulation(domain, initial_refines);
  auto B = Time::Bases();

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
  else if (problem == "cylinder")
    problem_data = CylinderProblem();
  else if (problem == "moving-peak")
    problem_data = MovingPeakProblem(vec_Xd);
  else {
    std::cout << "problem not recognized :-(" << std::endl;
    return 1;
  }

  AdaptiveHeatEquation heat_eq(vec_Xd, std::move(problem_data.first),
                               std::move(problem_data.second), adapt_opts);

  size_t ndof_Xd = 0;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(vec_Xd->container().size());
  double t_delta = heat_eq.Estimate(x0).second.second.error;
  std::cout << "t_init: " << t_delta << std::endl;
  size_t iter = 0;
  auto start_algorithm = std::chrono::steady_clock::now();
  while (ndof_Xd < max_dofs) {
    // Store a vector of all the nodes having maximum gradedness;
    std::vector<typename HeatEquation::TypeXVector::DNType*> max_gradedness;

    // A slight overestimate.
    ndof_Xd = vec_Xd->Bfs().size();
    size_t ndof_Xdd = heat_eq.vec_Xdd()->Bfs().size();
    size_t ndof_Ydd = heat_eq.vec_Ydd()->Bfs().size();
    std::cout << "iter: " << ++iter << "\n\tXDelta-size: " << ndof_Xd
              << "\n\tXDelta-Gradedness: "
              << vec_Xd->Gradedness(&max_gradedness)
              << "\n\tXDeltaDelta-size: " << ndof_Xdd
              << "\n\tYDeltaDelta-size: " << ndof_Ydd
              << "\n\ttotal-memory-kB: " << getmem() << std::flush;

    if (print_sampling) {
      auto sampling = PrintSampling(vec_Xd.get());
      std::cout << "\n\tsampling: ";
      for (auto [t, x, y, val] : sampling)
        std::cout << "" << t << "," << x << "," << y << "," << val << ";";
      std::cout << std::endl;
    }

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
                << "\n\tcond-time: " << duration_cond.count() << std::flush;
    }

    Eigen::VectorXd solution = x0;
    double total_error;
    AdaptiveHeatEquation::TypeXVector* residual;
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

    auto start_solve_estimate = std::chrono::steady_clock::now();
    do {
      t_delta /= adapt_opts.solve_factor;
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
                << "\n\t\tsolve-PCG-algebraic-error: "
                << pcg_data.algebraic_error
                << "\n\t\tsolve-time: " << duration_solve.count()
                << "\n\t\tsolve-memory: " << getmem() << std::flush;

      // Estimate.
      start = std::chrono::steady_clock::now();
      auto [residual, global_errors] = heat_eq.Estimate(solution);
      auto [residual_norm, global_error] = global_errors;
      total_error = global_error.error;
      std::chrono::duration<double> duration_estimate =
          std::chrono::steady_clock::now() - start;

      std::cout << "\n\t\tresidual-norm: " << residual_norm
                << "\n\t\testimate-time: " << duration_estimate.count()
                << "\n\t\testimate-memory: " << getmem() << std::flush;
      std::cout << "\n\t\tglobal-error: " << total_error
                << "\n\t\tYnorm-error: " << global_error.error_Yprime
                << "\n\t\tT0-error: " << global_error.error_t0 << std::flush;
      cycle++;
    } while (t_delta > adapt_opts.solve_xi * total_error);
    t_delta = total_error;

    std::chrono::duration<double> duration_solve_estimate =
        std::chrono::steady_clock::now() - start_solve_estimate;
    std::cout << "\n\tsolve-estimate-time: " << duration_solve_estimate.count();

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

    if (print_centers) {
      vec_Xd->FromVectorContainer(solution);
      auto print_dblnode = [](auto dblnode) {
        std::cout << "((" << dblnode->node_0()->level() << ","
                  << dblnode->node_0()->center() << "),"
                  << "(" << dblnode->node_1()->level() << ",("
                  << dblnode->node_1()->center().first << ","
                  << dblnode->node_1()->center().second
                  << ")) : " << dblnode->value() << ";";
      };

      std::cout << "\n\tcenters: ";
      for (auto dblnode : vec_Xd->Bfs()) print_dblnode(dblnode);

      std::cout << "\n\tcenters-max-gradedness: ";
      for (auto dblnode : max_gradedness) print_dblnode(dblnode);
    }

    if (print_time_slices.size()) {
      vec_Xd->FromVectorContainer(solution);
      for (double t : print_time_slices) {
        assert(t >= 0 && t <= 1);
        std::cerr << "time_slice " << t << " = ";
        PrintTimeSliceSS(t, vec_Xd.get());
        std::cerr << std::endl;
      }
      std::cerr << std::endl;
    }

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

    start = std::chrono::steady_clock::now();
    vec_Xd->FromVectorContainer(solution);
    auto r_info = heat_eq.Refine(marked_nodes);
    x0 = vec_Xd->ToVectorContainer();
    std::chrono::duration<double> duration_refine =
        std::chrono::steady_clock::now() - start;

    std::cout << "\n\tnodes-marked: " << r_info.nodes_marked
              << "\n\tnodes-conforming: " << r_info.nodes_conforming
              << "\n\tresidual-norm-marked: " << r_info.res_norm_marked
              << "\n\tresidual-norm-conforming: " << r_info.res_norm_conforming
              << "\n\trefine-time: " << duration_refine.count()
              << "\n\ttotal-time-algorithm: "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start_algorithm)
                     .count()
              << std::endl;
  }

  return 0;
}
