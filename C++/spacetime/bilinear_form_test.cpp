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
    ip += IntegrationRule<1, 2>::Integrate(
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
          typename OperatorSpace, typename BasisTimeIn,
          typename BasisTimeOut = BasisTimeIn>
void TestSpacetimeQuadrature(
    DoubleTreeVector<BasisTimeIn, HierarchicalBasisFn> &vec_in,
    DoubleTreeVector<BasisTimeOut, HierarchicalBasisFn> &vec_out,
    bool deriv_space = false, bool deriv_time_in = false,
    bool deriv_time_out = false) {
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
TEST(BilinearForm, XDeltaYDeltaFullTensorSparse) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  ortho_tree.UniformRefine(6);
  three_point_tree.UniformRefine(6);

  for (int level = 0; level < 6; level++) {
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

    TestSpacetimeQuadrature<Time::MassOperator, space::StiffnessOperator,
                            ThreePointWaveletFn, ThreePointWaveletFn>(
        vec_X_in, vec_X_out, /* deriv_space */ true, /* deriv_time_in */ false,
        /* deriv_time_out*/ false);
  }
}
}  // namespace spacetime
