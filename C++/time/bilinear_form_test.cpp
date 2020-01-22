#include "bilinear_form.hpp"

#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <cstdlib>
#include <set>
#include <unordered_map>

#include "../datastructures/multi_tree_vector.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "linear_operator.hpp"
#include "three_point_basis.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

namespace Time {
using datastructures::TreeVector;
using datastructures::TreeView;
using ::testing::ElementsAre;

template <typename BilinearForm, typename WaveletBasisIn>
void TestLinearity(BilinearForm bil_form,
                   const TreeView<WaveletBasisIn>& view_in) {
  // Create two random vectors.
  auto vec_in_1 =
      view_in.template DeepCopy<datastructures::TreeVector<WaveletBasisIn>>(
          [](auto nv, auto _) {
            nv->set_value(((double)std::rand()) / RAND_MAX);
          });
  auto vec_in_2 =
      view_in.template DeepCopy<datastructures::TreeVector<WaveletBasisIn>>(
          [](auto nv, auto _) {
            nv->set_value(((double)std::rand()) / RAND_MAX);
          });

  // Also calculate a lin. comb. of this vector
  double alpha = 1.337;
  auto vec_in_comb = vec_in_1.DeepCopy();
  vec_in_comb *= alpha;
  vec_in_comb += vec_in_2;

  // Apply this weighted comb. by hand
  bil_form.Apply(vec_in_1);
  auto vec_out_test = bil_form.vec_out()->DeepCopy();
  vec_out_test *= alpha;
  bil_form.Apply(vec_in_2);
  vec_out_test += *bil_form.vec_out();

  // Do it using the lin comb.
  bil_form.Apply(vec_in_comb);
  auto vec_out_comb = bil_form.vec_out()->DeepCopy();

  // Now check the results!
  auto nodes_comb = vec_out_comb.Bfs();
  auto nodes_test = vec_out_test.Bfs();
  ASSERT_GT(nodes_comb.size(), 0);
  ASSERT_EQ(nodes_comb.size(), nodes_test.size());
  for (int i = 0; i < nodes_comb.size(); ++i)
    ASSERT_NEAR(nodes_comb[i]->value(), nodes_test[i]->value(), 1e-10);
}

TEST(BilinearForm, ThreePointMass) {
  // Reset the persistent trees.
  ResetTrees();
  int ml = 7;
  three_point_tree.UniformRefine(ml);

  for (size_t j = 0; j < 50; ++j) {
    auto view_in = TreeView<ThreePointWaveletFn>(three_point_tree.meta_root);
    auto vec_out = TreeVector<ThreePointWaveletFn>(three_point_tree.meta_root);

    view_in.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });
    vec_out.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });

    BilinearForm<MassOperator, ThreePointWaveletFn, ThreePointWaveletFn>
        bil_form(&vec_out);

    // Test liniearity
    TestLinearity(bil_form, view_in);

    // Validate the result using quadrature on a matrix.
    auto mat = bil_form.ToMatrix(view_in);
    auto nodes_in = view_in.Bfs();
    auto nodes_out = vec_out.Bfs();
    for (int j = 0; j < nodes_in.size(); ++j)
      for (int i = 0; i < nodes_out.size(); ++i) {
        auto f = nodes_in[j]->node();
        auto g = nodes_out[i]->node();
        if (g->level() > f->level()) std::swap(f, g);

        double ip = 0;
        auto eval = [f, g](const double& t) { return f->Eval(t) * g->Eval(t); };
        for (auto elem : f->support())
          ip += boost::math::quadrature::gauss<double, 7>::integrate(
              eval, elem->Interval().first, elem->Interval().second);

        ASSERT_NEAR(mat(i, j), ip, 1e-10);
      }
  }
}

}  // namespace Time
