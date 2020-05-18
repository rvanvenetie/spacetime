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
  bool estimate_global_error = true;
  bool sparse_refine = true;
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
      "estimate_global_error", po::value<bool>(&estimate_global_error))(
      "sparse_refine", po::value<bool>(&sparse_refine));
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
    B.ortho_tree.UniformRefine(level + 2);
    B.three_point_tree.UniformRefine(level + 2);
    T.hierarch_basis_tree.UniformRefine(2 * (level + 2));

    auto vec_Xd = std::make_shared<
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    if (sparse_refine)
      vec_Xd->SparseRefine(level, {2, 1});
    else
      vec_Xd->UniformRefine({level, 2 * level});
    size_t ndof = vec_Xd->Bfs().size();  // A slight overestimate.
    std::cout << ndof << std::endl;
    if (ndof == 0) continue;
    if (ndof > max_dofs) break;
    AdaptiveHeatEquationOptions adapt_opts;
    adapt_opts.use_cache = false;
    AdaptiveHeatEquation heat_eq(vec_Xd, std::move(problem_data.first),
                                 std::move(problem_data.second), adapt_opts);
    std::cout << "XDelta-size: " << ndof << " total-memory-kB: " << getmem()
              << std::flush;

    // Solve - estimate.
    auto start = std::chrono::steady_clock::now();
    auto [solution, pcg_data] = heat_eq.Solve();
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
    std::cout << std::endl;
  }

  return 0;
}
