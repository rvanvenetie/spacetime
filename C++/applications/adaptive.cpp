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
  datastructures::TreeVector<HierarchicalBasisFn> time_slice(
      solution->root()->node_1());
  for (auto psi_time : solution->Project_0()->Bfs())
    if (psi_time->node()->Eval(t) != 0) {
      double time_val = psi_time->node()->Eval(t);
      // time_slice += time_val * psi_time->FrozenOtherAxis()
      time_slice.root()->Union(
          psi_time->FrozenOtherAxis(),
          /* call_filter*/ datastructures::func_true, /* call_postprocess*/
          [time_val](const auto& my_node, const auto& other_node) {
            my_node->set_value(my_node->value() +
                               time_val * other_node->value());
          });
    }

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
  size_t max_dofs = 0;
  bool estimate_global_error = true;
  bool calculate_condition_numbers = false;
  bool print_centers = false;
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
      "estimate_global_error", po::value<bool>(&estimate_global_error))(
      "calculate_condition_numbers",
      po::value<bool>(&calculate_condition_numbers))(
      "print_centers", po::value<bool>(&print_centers))(
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
  std::cout << "Problem options:" << std::endl;
  std::cout << "\tProblem: " << problem << std::endl;
  std::cout << "\tDomain: " << domain
            << "; initial-refines: " << initial_refines << std::endl;
  std::cout << std::endl;
  std::cout << adapt_opts << std::endl;

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

  auto T = InitialTriangulation(domain, initial_refines);
  auto B = Time::Bases();

  T.hierarch_basis_tree.UniformRefine(1);
  B.ortho_tree.UniformRefine(1);
  B.three_point_tree.UniformRefine(1);

  auto vec_Xd = std::make_shared<
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  vec_Xd->SparseRefine(1);

  AdaptiveHeatEquation heat_eq(vec_Xd, std::move(problem_data.first),
                               std::move(problem_data.second), adapt_opts);

  size_t ndof_X = 0, ndof_Y = 0;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(vec_Xd->container().size());
  size_t iter = 0;
  while (ndof_X < max_dofs) {
    ndof_X = vec_Xd->Bfs().size();             // A slight overestimate.
    ndof_Y = heat_eq.vec_Ydd()->Bfs().size();  // A slight overestimate.
    std::cout << "iter: " << ++iter << "\n\tXDelta-size: " << ndof_X
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
                << "\n\tcond-time: " << duration_cond.count() << std::flush;
    }

    // Solve.
    auto start = std::chrono::steady_clock::now();
    auto [solution, pcg_data] = heat_eq.Solve(x0);
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

    if (print_centers) {
      vec_Xd->FromVectorContainer(solution);
      for (auto dblnode : vec_Xd->Bfs()) {
        std::cerr << "((" << dblnode->node_0()->level() << ","
                  << dblnode->node_0()->center() << "),"
                  << "(" << dblnode->node_1()->level() << ",("
                  << dblnode->node_1()->center().first << ","
                  << dblnode->node_1()->center().second
                  << ")) : " << dblnode->value() << ";";
      }
      std::cerr << std::endl;
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

    // Estimate.
    if (estimate_global_error) {
      start = std::chrono::steady_clock::now();
      auto [global_error, terms] = heat_eq.EstimateGlobalError(solution);
      std::chrono::duration<double> duration_global =
          std::chrono::steady_clock::now() - start;
      std::cout << "\n\tglobal-error: " << global_error
                << "\n\tYnorm-error: " << terms.first
                << "\n\tT0-error: " << terms.second
                << "\n\tglobal-time: " << duration_global.count() << std::flush;
    }
    start = std::chrono::steady_clock::now();
    auto [residual, residual_norm] = heat_eq.Estimate(solution);
    std::chrono::duration<double> duration_estimate =
        std::chrono::steady_clock::now() - start;

    std::cout << "\n\tresidual-norm: " << residual_norm
              << "\n\testimate-time: " << duration_estimate.count()
              << "\n\testimate-memory: " << getmem() << std::flush;

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

    std::cout << "\n\trefine-time: " << duration_refine.count() << std::endl;
  }

  return 0;
}
