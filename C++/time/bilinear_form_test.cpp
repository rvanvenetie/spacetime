#include "bilinear_form.hpp"

#include <cmath>
#include <cstdlib>
#include <set>
#include <unordered_map>

#include "../datastructures/multi_tree_vector.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "integration.hpp"
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
using ::testing::ElementsAre;

template <template <typename, typename> class Operator, typename WaveletBasisIn,
          typename WaveletBasisOut>
void TestLinearity(const TreeVector<WaveletBasisIn>& vec_in,
                   const TreeVector<WaveletBasisOut>& vec_out) {
  // Create two random vectors.
  auto vec_in_1 = vec_in.DeepCopy();
  auto vec_in_2 = vec_in.DeepCopy();
  for (auto nv : vec_in_1.Bfs()) nv->set_random();
  for (auto nv : vec_in_2.Bfs()) nv->set_random();

  // Also calculate a lin. comb. of this vector
  double alpha = 1.337;
  auto vec_in_comb = vec_in_1.DeepCopy();
  vec_in_comb *= alpha;
  vec_in_comb += vec_in_2;

  // Apply this weighted comb. by hand
  CreateBilinearForm<Operator>(vec_in_1, vec_out).Apply();
  auto vec_out_test = vec_out.DeepCopy();
  vec_out_test *= alpha;
  CreateBilinearForm<Operator>(vec_in_2, vec_out).Apply();
  vec_out_test += vec_out;

  // Do it using the lin comb.
  CreateBilinearForm<Operator>(vec_in_comb, vec_out).Apply();
  auto vec_out_comb = vec_out.DeepCopy();

  // Now check the results!
  auto nodes_comb = vec_out_comb.Bfs();
  auto nodes_test = vec_out_test.Bfs();
  ASSERT_GT(nodes_comb.size(), 0);
  ASSERT_EQ(nodes_comb.size(), nodes_test.size());
  for (int i = 0; i < nodes_comb.size(); ++i)
    ASSERT_NEAR(nodes_comb[i]->value(), nodes_test[i]->value(), 1e-10);
}

template <template <typename, typename> class Operator, typename WaveletBasisIn,
          typename WaveletBasisOut>
void TestUppLow(const TreeVector<WaveletBasisIn>& vec_in,
                const TreeVector<WaveletBasisOut>& vec_out) {
  // Checks that BilForm::Apply() == BilForm::ApplyUpp() + BilForm::ApplyLow().
  for (int i = 0; i < 20; i++) {
    for (auto nv : vec_in.Bfs()) nv->set_random();

    CreateBilinearForm<Operator>(vec_in, vec_out).Apply();
    auto vec_out_full = vec_out.DeepCopy();

    CreateBilinearForm<Operator>(vec_in, vec_out).ApplyUpp();
    auto vec_out_upplow = vec_out.DeepCopy();
    CreateBilinearForm<Operator>(vec_in, vec_out).ApplyLow();
    vec_out_upplow += vec_out;

    // Now check the results!
    auto nodes_full = vec_out_full.Bfs();
    auto nodes_upplow = vec_out_upplow.Bfs();
    ASSERT_GT(nodes_full.size(), 0);
    ASSERT_EQ(nodes_full.size(), nodes_upplow.size());
    for (int i = 0; i < nodes_full.size(); ++i)
      ASSERT_NEAR(nodes_full[i]->value(), nodes_upplow[i]->value(), 1e-10);
  }
}

template <template <typename, typename> class Operator, typename WaveletBasisIn,
          typename WaveletBasisOut>
void CheckMatrixQuadrature(const TreeVector<WaveletBasisIn>& vec_in,
                           bool deriv_in, TreeVector<WaveletBasisOut>& vec_out,
                           bool deriv_out) {
  TestLinearity<Operator>(vec_in, vec_out);
  TestUppLow<Operator>(vec_in, vec_out);

  auto bil_form = CreateBilinearForm<Operator>(vec_in, vec_out);
  auto mat = bil_form.ToMatrix();
  auto nodes_in = vec_in.Bfs();
  auto nodes_out = vec_out.Bfs();
  for (int j = 0; j < nodes_in.size(); ++j)
    for (int i = 0; i < nodes_out.size(); ++i) {
      auto f = nodes_in[j]->node();
      auto g = nodes_out[i]->node();
      auto support = f->support();
      if (g->level() > f->level()) support = g->support();
      double ip = 0;
      for (auto elem : support)
        ip += Integrate(
            [f, deriv_in, g, deriv_out](const double& t) {
              return f->Eval(t, deriv_in) * g->Eval(t, deriv_out);
            },
            *elem, /*degree*/ 2);

      ASSERT_NEAR(mat(i, j), ip, 1e-10);
    }

  // Check that its transpose equals the matrix transpose.
  auto tmat = bil_form.Transpose().ToMatrix();
  ASSERT_TRUE(mat.transpose().isApprox(tmat));
}

TEST(BilinearForm, FullTest) {
  // Reset the persistent trees.
  ResetTrees();
  int ml = 7;
  three_point_tree.UniformRefine(ml);
  ortho_tree.UniformRefine(ml);

  for (size_t j = 0; j < 20; ++j) {
    // Set up three-point tree.
    auto three_vec_in =
        TreeVector<ThreePointWaveletFn>(three_point_tree.meta_root);
    auto three_vec_out =
        TreeVector<ThreePointWaveletFn>(three_point_tree.meta_root);
    three_vec_in.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });
    three_vec_out.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });
    ASSERT_GT(three_vec_in.Bfs().size(), 0);
    ASSERT_GT(three_vec_out.Bfs().size(), 0);

    // Set up orthonormal tree.
    auto ortho_vec_in = TreeVector<OrthonormalWaveletFn>(ortho_tree.meta_root);
    auto ortho_vec_out = TreeVector<OrthonormalWaveletFn>(ortho_tree.meta_root);
    ortho_vec_in.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 == 0;
        });
    ortho_vec_out.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 == 0;
        });
    ASSERT_GT(ortho_vec_in.Bfs().size(), 0);
    ASSERT_GT(ortho_vec_out.Bfs().size(), 0);

    TestLinearity<ZeroEvalOperator>(three_vec_in, three_vec_out);
    TestUppLow<ZeroEvalOperator>(three_vec_in, three_vec_out);

    // Test linearity and validate the result using quadrature on a matrix.
    CheckMatrixQuadrature<MassOperator, ThreePointWaveletFn,
                          ThreePointWaveletFn>(three_vec_in, false,
                                               three_vec_out, false);
    CheckMatrixQuadrature<MassOperator, OrthonormalWaveletFn,
                          OrthonormalWaveletFn>(ortho_vec_in, false,
                                                ortho_vec_out, false);
    CheckMatrixQuadrature<MassOperator, OrthonormalWaveletFn,
                          ThreePointWaveletFn>(ortho_vec_in, false,
                                               three_vec_out, false);
    CheckMatrixQuadrature<MassOperator, ThreePointWaveletFn,
                          OrthonormalWaveletFn>(three_vec_in, false,
                                                ortho_vec_out, false);
    CheckMatrixQuadrature<TransportOperator, ThreePointWaveletFn,
                          OrthonormalWaveletFn>(three_vec_in, true,
                                                ortho_vec_out, false);
  }
}

}  // namespace Time
