#include <cmath>
#include <map>
#include <random>
#include <set>

#include "time/basis.hpp"
#include "time/bilinear_form.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

using namespace Time;
using namespace datastructures;

constexpr int level = 10;
constexpr int bilform_iters = 10;
constexpr int inner_iters = 150;

int main() {
  ortho_tree.UniformRefine(::level);

  for (size_t j = 0; j < ::bilform_iters; ++j) {
    // Set up ortho tree.
    auto vec_in = TreeVector<OrthonormalWaveletFn>(ortho_tree.meta_root);
    auto vec_out = TreeVector<OrthonormalWaveletFn>(ortho_tree.meta_root);
    vec_in.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });
    vec_out.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });

    BilinearForm<MassOperator, OrthonormalWaveletFn, OrthonormalWaveletFn>
        bil_form(&vec_out);
    for (size_t k = 0; k < ::inner_iters; k++) {
      vec_out.Reset();
      for (auto& nv : vec_in.Bfs()) {
        nv->set_value(bsd_rnd());
      }
      bil_form.Apply(vec_in);
    }
  }
  return 0;
}
