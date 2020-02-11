#include "bilinear_form.hpp"

#include "../space/initial_triangulation.hpp"
#include "../space/operators.hpp"
#include "../time/linear_operator.hpp"
#include "../tools/integration.hpp"
#include "basis.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::ortho_tree;
using Time::OrthonormalWaveletFn;
using Time::three_point_tree;
using Time::ThreePointWaveletFn;
using tools::IntegrationRule;

namespace spacetime {

template <typename WaveletBasisIn, typename WaveletBasisOut>
double TimeQuadrature(WaveletBasisIn *f, WaveletBasisOut *g, bool deriv_in,
                      bool deriv_out) {
  auto support = f->support();
  if (g->level() > f->level()) support = g->support();
  double ip = 0;
  for (auto elem : support)
    ip += IntegrationRule</*dim*/ 1, /*degree*/ 2>::Integrate(
        [f, deriv_in, g, deriv_out](const double &t) {
          return f->Eval(t, deriv_in) * g->Eval(t, deriv_out);
        },
        *elem);
  return ip;
}

double SpaceQuadrature(HierarchicalBasisFn *fn_in, HierarchicalBasisFn *fn_out,
                       bool deriv) {
  double quad = 0;
  if (!fn_out->vertex()->on_domain_boundary &&
      !fn_in->vertex()->on_domain_boundary) {
    auto elems_fine =
        fn_in->level() < fn_out->level() ? fn_out->support() : fn_in->support();
    for (auto elem : elems_fine) {
      if (deriv) {
        quad += IntegrationRule</*dim*/ 2, /*degree*/ 0>::Integrate(
            [&](double x, double y) {
              return fn_out->EvalGrad(x, y).dot(fn_in->EvalGrad(x, y));
            },
            *elem);
      } else {
        quad += IntegrationRule</*dim*/ 2, /*degree*/ 2>::Integrate(
            [&](double x, double y) {
              return fn_out->Eval(x, y) * fn_in->Eval(x, y);
            },
            *elem);
      }
    }
  }
  return quad;
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
void TestSpacetimeCache(
    DoubleTreeVector<BasisTimeIn, HierarchicalBasisFn> &vec_in,
    DoubleTreeVector<BasisTimeOut, HierarchicalBasisFn> &vec_out) {
  for (int i = 0; i < 2; ++i) {
    // Create a bilinear form and do some caching tests.
    auto bil_form = CreateBilinearForm<OperatorTime, OperatorSpace>(
        vec_in, &vec_out, /* use_cache */ i == 0);

    // Put some random values into vec_in.
    for (auto nv : vec_in.Bfs())
      nv->set_value(((double)std::rand()) / RAND_MAX);

    // Apply the spacetime bilinear form.
    bil_form.Apply();

    // Check that applying it a couple times still gives the same result.
    auto eigen_out = vec_out.ToVector();
    bil_form.Apply();
    bil_form.Apply();
    ASSERT_TRUE(eigen_out.isApprox(vec_out.ToVector()));

    // Check that applying it with a different input gives another result.
    for (auto nv : vec_in.Bfs())
      nv->set_value(((double)std::rand()) / RAND_MAX);
    bil_form.Apply();
    ASSERT_FALSE(eigen_out.isApprox(vec_out.ToVector()));

    // Which stays the same if we apply it multiple times.
    eigen_out = vec_out.ToVector();
    bil_form.Apply();
    bil_form.Apply();
    ASSERT_TRUE(eigen_out.isApprox(vec_out.ToVector()));
  }
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
void TestSpacetimeLinearity(
    DoubleTreeVector<BasisTimeIn, HierarchicalBasisFn> &vec_in,
    DoubleTreeVector<BasisTimeOut, HierarchicalBasisFn> &vec_out) {
  // Create two random vectors.
  auto vec_in_1 = vec_in.DeepCopy();
  auto vec_in_2 = vec_in.DeepCopy();
  for (auto nv : vec_in_1.Bfs())
    nv->set_value(((double)std::rand()) / RAND_MAX);
  for (auto nv : vec_in_1.Bfs())
    nv->set_value(((double)std::rand()) / RAND_MAX);

  // Also calculate a lin. comb. of this vector
  double alpha = 1.337;
  auto vec_in_comb = vec_in_1.DeepCopy();
  vec_in_comb *= alpha;
  vec_in_comb += vec_in_2;

  // Apply this weighted comb. by hand
  CreateBilinearForm<OperatorTime, OperatorSpace>(vec_in_1, &vec_out).Apply();
  auto vec_out_test = vec_out.DeepCopy();
  vec_out_test *= alpha;
  CreateBilinearForm<OperatorTime, OperatorSpace>(vec_in_2, &vec_out).Apply();
  vec_out_test += vec_out;

  // Do it using the lin comb.
  CreateBilinearForm<OperatorTime, OperatorSpace>(vec_in_comb, &vec_out)
      .Apply();
  auto vec_out_comb = vec_out.DeepCopy();

  // Now check the results!
  auto nodes_comb = vec_out_comb.Bfs();
  auto nodes_test = vec_out_test.Bfs();
  ASSERT_GT(nodes_comb.size(), 0);
  ASSERT_EQ(nodes_comb.size(), nodes_test.size());
  for (int i = 0; i < nodes_comb.size(); ++i)
    ASSERT_NEAR(nodes_comb[i]->value(), nodes_test[i]->value(), 1e-10);
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn,
          typename BasisTimeOut = BasisTimeIn>
void TestSpacetimeQuadrature(
    DoubleTreeVector<BasisTimeIn, HierarchicalBasisFn> &vec_in,
    DoubleTreeVector<BasisTimeOut, HierarchicalBasisFn> &vec_out,
    bool deriv_space = false, bool deriv_time_in = false,
    bool deriv_time_out = false) {
  // First do some linearity check!
  TestSpacetimeLinearity<OperatorTime, OperatorSpace, BasisTimeIn,
                         BasisTimeOut>(vec_in, vec_out);

  // Then do some cache checks!
  TestSpacetimeCache<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>(
      vec_in, vec_out);

  // Create a bilinear form and do some quadrature tests.
  auto bil_form =
      CreateBilinearForm<OperatorTime, OperatorSpace>(vec_in, &vec_out);

  // Simply put some random values into vec_in.
  for (auto nv : vec_in.Bfs()) nv->set_value(((double)std::rand()) / RAND_MAX);

  // Apply the spacetime bilinear form.
  bil_form.Apply();

  // Now compare this to the matrix approach
  auto db_nodes_in = vec_in.Bfs();
  auto db_nodes_out = vec_out.Bfs();

  for (int j = 0; j < db_nodes_out.size(); ++j) {
    double quad_val = 0;
    for (int i = 0; i < db_nodes_in.size(); ++i) {
      quad_val +=
          db_nodes_in[i]->value() *
          TimeQuadrature(std::get<0>(db_nodes_in[i]->nodes()),
                         std::get<0>(db_nodes_out[j]->nodes()), deriv_time_in,
                         deriv_time_out) *
          SpaceQuadrature(std::get<1>(db_nodes_in[i]->nodes()),
                          std::get<1>(db_nodes_out[j]->nodes()), deriv_space);
    }
    ASSERT_NEAR(quad_val, db_nodes_out[j]->value(), 1e-10);
  }
}

TEST(BilinearForm, SparseQuadrature) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  ortho_tree.UniformRefine(6);
  three_point_tree.UniformRefine(6);

  for (int level = 1; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.SparseRefine(level);
    auto Y_delta = GenerateYDelta(X_delta);

    auto vec_X_in = X_delta.template DeepCopy<
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>();
    auto vec_X_out = X_delta.template DeepCopy<
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>();
    auto vec_Y_in = Y_delta.template DeepCopy<
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>();
    auto vec_Y_out = Y_delta.template DeepCopy<
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>();

    // Test the actual operators that we use.
    TestSpacetimeQuadrature<Time::MassOperator, space::StiffnessOperator,
                            ThreePointWaveletFn, ThreePointWaveletFn>(
        vec_X_in, vec_X_out, /* deriv_space */ true,
        /* deriv_time_in */ false,
        /* deriv_time_out*/ false);
    TestSpacetimeQuadrature<Time::TransportOperator, space::MassOperator,
                            ThreePointWaveletFn, OrthonormalWaveletFn>(
        vec_X_in, vec_Y_out, /* deriv_space */ false,
        /* deriv_time_in */ true,
        /* deriv_time_out*/ false);
    TestSpacetimeQuadrature<Time::TransportOperator, space::StiffnessOperator,
                            ThreePointWaveletFn, OrthonormalWaveletFn>(
        vec_X_in, vec_Y_out, /* deriv_space */ true,
        /* deriv_time_in */ true,
        /* deriv_time_out*/ false);

    // Test some stuff we *could* use.
    TestSpacetimeQuadrature<Time::MassOperator, space::MassOperator,
                            OrthonormalWaveletFn, OrthonormalWaveletFn>(
        vec_Y_in, vec_Y_out, /* deriv_space */ false,
        /* deriv_time_in */ false,
        /* deriv_time_out*/ false);
    TestSpacetimeQuadrature<Time::MassOperator, space::MassOperator,
                            OrthonormalWaveletFn, ThreePointWaveletFn>(
        vec_Y_in, vec_X_out, /* deriv_space */ false,
        /* deriv_time_in */ false,
        /* deriv_time_out*/ false);
  }
}
}  // namespace spacetime
