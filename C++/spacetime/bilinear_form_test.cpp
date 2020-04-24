#include "bilinear_form.hpp"

#include "basis.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "space/integration.hpp"
#include "space/operators.hpp"
#include "time/bases.hpp"
#include "time/integration.hpp"
#include "time/linear_operator.hpp"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

namespace spacetime {

template <typename BilForm>
Eigen::MatrixXd ToMatrix(BilForm &bilform) {
  auto nodes_in = bilform.vec_in()->Bfs();
  auto nodes_out = bilform.vec_out()->Bfs();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nodes_out.size(), nodes_in.size());
  for (int i = 0; i < nodes_in.size(); ++i) {
    bilform.vec_in()->Reset();
    if (!std::get<1>(nodes_in[i]->nodes())->on_domain_boundary())
      nodes_in[i]->set_value(1);
    bilform.vec_out()->FromVectorContainer(
        bilform.Apply(bilform.vec_in()->ToVectorContainer()));
    for (int j = 0; j < nodes_out.size(); ++j) {
      A(j, i) = nodes_out[j]->value();
    }
  }
  return A;
}

template <typename WaveletBasisIn, typename WaveletBasisOut>
double TimeQuadrature(WaveletBasisIn *f, WaveletBasisOut *g, bool deriv_in,
                      bool deriv_out) {
  auto support = f->support();
  if (g->level() > f->level()) support = g->support();
  double ip = 0;
  for (auto elem : support)
    ip += Time::Integrate(
        [f, deriv_in, g, deriv_out](const double &t) {
          return f->Eval(t, deriv_in) * g->Eval(t, deriv_out);
        },
        *elem, /*degree*/ 2);
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
        quad += space::Integrate(
            [&](double x, double y) {
              return fn_out->EvalGrad(x, y).dot(fn_in->EvalGrad(x, y));
            },
            *elem, /*degree*/ 0);
      } else {
        quad += space::Integrate(
            [&](double x, double y) {
              return fn_out->Eval(x, y) * fn_in->Eval(x, y);
            },
            *elem, /*degree*/ 2);
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
        &vec_in, &vec_out, /* use_cache */ i == 0);

    // Put some random values into vec_in.
    for (auto nv : vec_in.Bfs()) {
      if (std::get<1>(nv->nodes())->on_domain_boundary()) continue;
      nv->set_random();
    }
    Eigen::VectorXd v_in = vec_in.ToVectorContainer();

    // Apply the spacetime bilinear form.
    auto eigen_out = bil_form->Apply(v_in);

    // Check that applying it a couple times still gives the same result.
    bil_form->Apply(v_in);
    ASSERT_TRUE(eigen_out.isApprox(bil_form->Apply(v_in)));

    // Check that applying it with a different input gives another result.
    for (auto nv : vec_in.Bfs()) {
      if (std::get<1>(nv->nodes())->on_domain_boundary()) continue;
      nv->set_random();
    }
    ASSERT_FALSE(
        eigen_out.isApprox(bil_form->Apply(vec_in.ToVectorContainer())));
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
  for (auto nv : vec_in_1.Bfs()) {
    if (std::get<1>(nv->nodes())->on_domain_boundary()) continue;
    nv->set_random();
  }
  for (auto nv : vec_in_2.Bfs()) {
    if (std::get<1>(nv->nodes())->on_domain_boundary()) continue;
    nv->set_random();
  }

  // Also calculate a lin. comb. of this vector
  double alpha = 1.337;
  auto vec_in_comb = vec_in_1.DeepCopy();
  vec_in_comb *= alpha;
  vec_in_comb += vec_in_2;

  // Apply this weighted comb. by hand
  auto vec_out_test =
      CreateBilinearForm<OperatorTime, OperatorSpace>(&vec_in_1, &vec_out,
                                                      /*use_cache*/ true)
          ->Apply(vec_in_1.ToVectorContainer());
  vec_out_test *= alpha;
  vec_out_test +=
      CreateBilinearForm<OperatorTime, OperatorSpace>(&vec_in_2, &vec_out,
                                                      /*use_cache*/ true)
          ->Apply(vec_in_2.ToVectorContainer());

  // Do it using the lin comb.
  auto vec_out_comb =
      CreateBilinearForm<OperatorTime, OperatorSpace>(&vec_in_comb, &vec_out,
                                                      /*use_cache*/ true)
          ->Apply(vec_in_comb.ToVectorContainer());

  // Now check the results!
  ASSERT_TRUE(vec_out_comb.isApprox(vec_out_test));
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
  auto bil_form = CreateBilinearForm<OperatorTime, OperatorSpace>(
      &vec_in, &vec_out, /*use_cache*/ true);

  // Simply put some random values into vec_in.
  for (auto nv : vec_in.Bfs()) {
    if (std::get<1>(nv->nodes())->on_domain_boundary()) continue;
    nv->set_random();
  }

  // Apply the spacetime bilinear form.
  auto v_in = vec_in.ToVectorContainer();
  auto v_out = bil_form->Apply(v_in);

  // Now compare this to the matrix approach
  const auto &db_nodes_in = vec_in.container();
  const auto &db_nodes_out = vec_out.container();
  for (int j = 0; j < vec_out.container().size(); ++j) {
    double quad_val = 0;
    if (db_nodes_out[j].is_metaroot()) continue;
    for (int i = 0; i < vec_in.container().size(); ++i) {
      if (db_nodes_in[i].is_metaroot()) continue;
      quad_val +=
          v_in[i] *
          TimeQuadrature(std::get<0>(db_nodes_in[i].nodes()),
                         std::get<0>(db_nodes_out[j].nodes()), deriv_time_in,
                         deriv_time_out) *
          SpaceQuadrature(std::get<1>(db_nodes_in[i].nodes()),
                          std::get<1>(db_nodes_out[j].nodes()), deriv_space);
    }
    ASSERT_NEAR(quad_val, v_out[j], 1e-10);
  }
}

TEST(BilinearForm, SparseQuadrature) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.ortho_tree.UniformRefine(6);
  B.three_point_tree.UniformRefine(6);

  for (int level = 1; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.SparseRefine(level);
    auto Y_delta = GenerateYDelta<DoubleTreeView>(X_delta);

    auto vec_X = X_delta.template DeepCopy<
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>();
    auto vec_Y = Y_delta.template DeepCopy<
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>();

    // Test the actual operators that we use.
    TestSpacetimeQuadrature<Time::MassOperator, space::StiffnessOperator,
                            OrthonormalWaveletFn, OrthonormalWaveletFn>(
        vec_Y, vec_Y, /* deriv_space */ true,
        /* deriv_time_in */ false,
        /* deriv_time_out*/ false);
    TestSpacetimeQuadrature<Time::TransportOperator, space::MassOperator,
                            ThreePointWaveletFn, OrthonormalWaveletFn>(
        vec_X, vec_Y, /* deriv_space */ false,
        /* deriv_time_in */ true,
        /* deriv_time_out*/ false);
    TestSpacetimeQuadrature<Time::MassOperator, space::StiffnessOperator,
                            ThreePointWaveletFn, OrthonormalWaveletFn>(
        vec_X, vec_Y, /* deriv_space */ true,
        /* deriv_time_in */ false,
        /* deriv_time_out*/ false);

    // Test some stuff we *could* use.
    TestSpacetimeQuadrature<Time::MassOperator, space::StiffnessOperator,
                            ThreePointWaveletFn, ThreePointWaveletFn>(
        vec_X, vec_X, /* deriv_space */ true,
        /* deriv_time_in */ false,
        /* deriv_time_out*/ false);
    TestSpacetimeQuadrature<Time::MassOperator, space::MassOperator,
                            OrthonormalWaveletFn, OrthonormalWaveletFn>(
        vec_Y, vec_Y, /* deriv_space */ false,
        /* deriv_time_in */ false,
        /* deriv_time_out*/ false);
    TestSpacetimeQuadrature<Time::MassOperator, space::MassOperator,
                            OrthonormalWaveletFn, ThreePointWaveletFn>(
        vec_Y, vec_X, /* deriv_space */ false,
        /* deriv_time_in */ false,
        /* deriv_time_out*/ false);
    TestSpacetimeQuadrature<Time::TransportOperator, space::StiffnessOperator,
                            ThreePointWaveletFn, OrthonormalWaveletFn>(
        vec_X, vec_Y, /* deriv_space */ true,
        /* deriv_time_in */ true,
        /* deriv_time_out*/ false);

    // Check that it also works for different vec_out.
    auto vec_Y_cpy = Y_delta.template DeepCopy<
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>();
    TestSpacetimeQuadrature<Time::MassOperator, space::StiffnessOperator,
                            OrthonormalWaveletFn, OrthonormalWaveletFn>(
        vec_Y, vec_Y_cpy, /* deriv_space */ true,
        /* deriv_time_in */ false,
        /* deriv_time_out*/ false);
  }
}

TEST(BilinearForm, Transpose) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.ortho_tree.UniformRefine(6);
  B.three_point_tree.UniformRefine(6);

  for (int level = 1; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.SparseRefine(level);
    auto Y_delta = GenerateYDelta<DoubleTreeView>(X_delta);

    auto vec_X = X_delta.template DeepCopy<
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>();
    auto vec_Y = Y_delta.template DeepCopy<
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>();

    // Test the actual operators that we use.
    auto A_s = CreateBilinearForm<Time::MassOperator, space::StiffnessOperator>(
        &vec_X, &vec_Y, /*use_cache*/ true);

    // Reuse theta/sigma for B_t.
    auto B_t = CreateBilinearForm<Time::TransportOperator, space::MassOperator>(
        &vec_X, &vec_Y, A_s->sigma(), A_s->theta(), /*use_cache*/ true);

    // Create matrices.
    auto mat_A_s = ToMatrix(*A_s);
    auto mat_B_t = ToMatrix(*B_t);

    // Create transpose.
    auto trans_A_s = A_s->Transpose();
    auto trans_B_t = B_t->Transpose();

    // Compare the matrices.
    ASSERT_TRUE(mat_A_s.transpose().isApprox(ToMatrix(*trans_A_s)));
    ASSERT_TRUE(mat_B_t.transpose().isApprox(ToMatrix(*trans_B_t)));

    // Now check the sum.
    auto B = SumBilinearForm(A_s, B_t);
    auto mat_B = ToMatrix(B);
    ASSERT_TRUE((mat_A_s + mat_B_t).isApprox(mat_B));

    // Now check the transpose of the sum.
    auto BT = B.Transpose();
    ASSERT_TRUE((mat_A_s + mat_B_t).transpose().isApprox(ToMatrix(*BT)));
  }
}

TEST(SymmetricBilinearForm, Works) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.ortho_tree.UniformRefine(6);
  B.three_point_tree.UniformRefine(6);

  for (int level = 1; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.SparseRefine(level);
    auto Y_delta = GenerateYDelta<DoubleTreeView>(X_delta);

    auto vec_X = X_delta.template DeepCopy<
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>();
    auto vec_Y = Y_delta.template DeepCopy<
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>();

    // Test the actual operators that we use.
    for (bool use_cache : {true, false}) {
      // A_s
      {
        BilinearForm<Time::MassOperator, space::StiffnessOperator,
                     OrthonormalWaveletFn, OrthonormalWaveletFn>
            A_normal(&vec_Y, &vec_Y, /* use_cache */ use_cache);
        SymmetricBilinearForm<Time::MassOperator, space::StiffnessOperator,
                              OrthonormalWaveletFn>
            A_symm(&vec_Y, /* use_cache */ use_cache);

        // Put some random values into vec_Y.
        for (auto nv : vec_Y.Bfs()) {
          if (std::get<1>(nv->nodes())->on_domain_boundary()) continue;
          nv->set_random();
        }
        Eigen::VectorXd v_in = vec_Y.ToVectorContainer();

        // Test that the results equal.
        ASSERT_TRUE(A_normal.Apply(v_in).isApprox(A_symm.Apply(v_in)));
      }

      // G
      {
        BilinearForm<Time::ZeroEvalOperator, space::MassOperator,
                     ThreePointWaveletFn, ThreePointWaveletFn>
            G_normal(&vec_X, &vec_X, /* use_cache */ use_cache);
        SymmetricBilinearForm<Time::ZeroEvalOperator, space::MassOperator,
                              ThreePointWaveletFn>
            G_symm(&vec_X, /* use_cache */ true);

        // Put some random values into vec_Y.
        for (auto nv : vec_X.Bfs()) {
          if (std::get<1>(nv->nodes())->on_domain_boundary()) continue;
          nv->set_random();
        }
        Eigen::VectorXd v_in = vec_X.ToVectorContainer();

        // Test that the results equal.
        ASSERT_TRUE(G_normal.Apply(v_in).isApprox(G_symm.Apply(v_in)));
      }
    }
  }
}

TEST(BlockDiagonalBilinearForm, CanBeConstructed) {
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  B.ortho_tree.UniformRefine(6);
  B.three_point_tree.UniformRefine(6);

  for (int level = 1; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.SparseRefine(level);
    auto Y_delta = GenerateYDelta<DoubleTreeView>(X_delta);

    auto vec_Y = Y_delta.template DeepCopy<
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>();

    auto A_s = CreateBlockDiagonalBilinearForm<space::StiffnessOperator>(
        &vec_Y, &vec_Y, /*use_cache*/ true);
    auto mat_A_s = ToMatrix(*A_s);

    auto P_Y = CreateBlockDiagonalBilinearForm<
        space::DirectInverse<space::StiffnessOperator>>(&vec_Y, &vec_Y,
                                                        /*use_cache*/ true);
    auto mat_P_Y = ToMatrix(*P_Y);

    ASSERT_TRUE((mat_A_s * mat_P_Y).isApprox(mat_P_Y * mat_A_s));
  }
}
}  // namespace spacetime
