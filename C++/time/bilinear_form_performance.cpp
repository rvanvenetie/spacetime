#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "../tools/util.hpp"
#include "bases.hpp"
#include "bilinear_form.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

using namespace Time;
using namespace datastructures;

namespace po = boost::program_options;

template <typename I>
long long MaxIndex(int level);

template <>
long long MaxIndex<OrthonormalWaveletFn>(int level) {
  return std::pow(2, level);
}

template <>
long long MaxIndex<ThreePointWaveletFn>(int level) {
  return std::pow(2, level - 1);
}

template <typename I>
TreeVector<I> GradedTree(I *meta_root, int max_lvl, bool towards_origin,
                         double theta = 0.5) {
  TreeVector<I> result(meta_root);
  result.DeepRefine(/* call_filter */
                    [&](auto psi) {
                      if (towards_origin)
                        return (psi->level() <= max_lvl &&
                                psi->index() <=
                                    std::pow(1 + theta, psi->level()));
                      else
                        return (psi->level() <= max_lvl &&
                                psi->index() >=
                                    MaxIndex<I>(psi->level()) -
                                        std::pow(1 + theta, psi->level()));
                    }, /* call_postprocess */
                    [&](auto nv) {
                      nv->set_random();
                      nv->node()->Refine();
                    });
  return result;
}
auto Now() { return std::chrono::high_resolution_clock::now(); }
double Duration(std::chrono::high_resolution_clock::time_point start) {
  return std::chrono::duration<double>(Now() - start).count();
}

template <template <typename, typename> class Operator, typename WaveletBasisIn,
          typename WaveletBasisOut>
void TimeBilForm(std::string name,
                 const datastructures::TreeVector<WaveletBasisIn> &vec_in,
                 const datastructures::TreeVector<WaveletBasisOut> &vec_out) {
  double time_create = 0, time_apply = 0, time_apply_upp = 0,
         time_apply_low = 0;
  size_t iters = 0;
  while (time_apply < 20 && iters < 30) {
    ++iters;
    auto time_start = Now();
    auto bilform = CreateBilinearForm<Operator>(vec_in, vec_out);
    time_create += Duration(time_start);

    time_start = Now();
    bilform.Apply();
    time_apply += Duration(time_start);

    time_start = Now();
    bilform.ApplyUpp();
    time_apply_upp += Duration(time_start);

    time_start = Now();
    bilform.ApplyLow();
    time_apply_low += Duration(time_start);
  }
  std::cout << "\n\ttime-" << name << "-create: " << time_create / iters;
  std::cout << "\n\ttime-" << name << "-apply: " << time_apply / iters;
  std::cout << "\n\ttime-" << name << "-apply-upp: " << time_apply_upp / iters;
  std::cout << "\n\ttime-" << name << "-apply-low: " << time_apply_low / iters;
  std::cout << std::flush;
}

int main(int argc, char *argv[]) {
  double theta = 0.3;
  bool print_mesh = false;
  size_t max_iter = 999;
  boost::program_options::options_description adapt_optdesc("Refine options");
  adapt_optdesc.add_options()("theta", po::value<double>(&theta))(
      "print_mesh", po::value<bool>(&print_mesh))("max_iter",
                                                  po::value<size_t>(&max_iter));
  boost::program_options::options_description cmdline_options;
  cmdline_options.add(adapt_optdesc);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(),
            vm);
  po::notify(vm);
  std::cout << "Adaptive options:" << std::endl;
  std::cout << "\ttheta: " << theta << std::endl << std::endl;

  Bases B;
  int iter = 0;
  bool uniform_refine = theta >= 1;
  while (iter < max_iter) {
    std::cout << "iter: " << ++iter;

    if (uniform_refine) {
      TreeVector<OrthonormalWaveletFn> ortho_tree(B.ortho_tree.meta_root());
      TreeVector<ThreePointWaveletFn> threept_tree(
          B.three_point_tree.meta_root());
      ortho_tree.UniformRefine(std::array{iter}, /* grow_tree */ true);
      threept_tree.UniformRefine(std::array{iter}, /* grow_tree */ true);
      std::cout << "\n\tortho-tree-size: " << ortho_tree.Bfs().size()
                << "\n\tthreept-tree-size: " << threept_tree.Bfs().size()
                << "\n\ttotal-memory-kB: " << getmem() << std::flush;
      TimeBilForm<MassOperator>("M-o-o", ortho_tree, ortho_tree);
      TimeBilForm<MassOperator>("M-t-o", threept_tree, ortho_tree);
      TimeBilForm<TransportOperator>("T-t-o", threept_tree, ortho_tree);
    } else {
      // Local refine.
      auto ortho_tree_0 =
          GradedTree(B.ortho_tree.meta_root(), iter, true, theta);
      auto ortho_tree_1 =
          GradedTree(B.ortho_tree.meta_root(), iter, false, theta);
      auto threept_tree_0 =
          GradedTree(B.three_point_tree.meta_root(), iter, true, theta);
      auto threept_tree_1 =
          GradedTree(B.three_point_tree.meta_root(), iter, false, theta);
      std::cout << "\n\tortho-tree-0-size: " << ortho_tree_0.Bfs().size()
                << "\n\tortho-tree-1-size: " << ortho_tree_1.Bfs().size()
                << "\n\tthreept-tree-0-size: " << threept_tree_0.Bfs().size()
                << "\n\tthreept-tree-1-size: " << threept_tree_1.Bfs().size()
                << "\n\ttotal-memory-kB: " << getmem() << std::flush;

      TimeBilForm<MassOperator>("M-o0-o0", ortho_tree_0, ortho_tree_0);
      // TimeBilForm<MassOperator>("M-o0-o1", ortho_tree_0, ortho_tree_1);
      // TimeBilForm<MassOperator>("M-t0-t0", threept_tree_0, threept_tree_0);
      // TimeBilForm<MassOperator>("M-t0-t1", threept_tree_0, threept_tree_1);
      // TimeBilForm<MassOperator>("M-o0-t0", ortho_tree_0, threept_tree_0);
      // TimeBilForm<MassOperator>("M-o0-t1", ortho_tree_0, threept_tree_1);
      TimeBilForm<MassOperator>("M-t0-o0", threept_tree_0, ortho_tree_0);
      TimeBilForm<MassOperator>("M-t0-o1", threept_tree_0, ortho_tree_1);

      TimeBilForm<TransportOperator>("T-t0-o0", threept_tree_0, ortho_tree_0);
      TimeBilForm<TransportOperator>("T-t0-o1", threept_tree_0, ortho_tree_1);
      // TimeBilForm<TransportOperator>("T-t1-o0", threept_tree_1,
      // ortho_tree_0);

      if (print_mesh) {
        std::cout << "\n\tortho-tree-0: [";
        for (auto nv : ortho_tree_0.Bfs())
          std::cout << "(" << nv->node()->level() << "," << nv->node()->center()
                    << "),";
        std::cout << "]" << std::flush;
        std::cout << "\n\tortho-tree-1: [";
        for (auto nv : ortho_tree_1.Bfs())
          std::cout << "(" << nv->node()->level() << "," << nv->node()->center()
                    << "),";
        std::cout << "]" << std::flush;
        std::cout << "\n\tthreept-tree-0: [";
        for (auto nv : threept_tree_0.Bfs())
          std::cout << "(" << nv->node()->level() << "," << nv->node()->center()
                    << "),";
        std::cout << "]" << std::flush;
        std::cout << "\n\tthreept-tree-1: [";
        for (auto nv : threept_tree_1.Bfs())
          std::cout << "(" << nv->node()->level() << "," << nv->node()->center()
                    << "),";
        std::cout << "]" << std::flush;
      }
    }
    std::cout << std::endl;
  }
  return 0;
}
