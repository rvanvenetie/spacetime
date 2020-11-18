#include "adaptive_heat_equation.hpp"

#include "../space/initial_triangulation.hpp"
#include "../time/basis.hpp"
#include "../tools/linalg.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "problems.hpp"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using spacetime::GenerateYDelta;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

namespace applications {

// Creates a slice in *space*.
template <typename BasisTime>
datastructures::TreeVector<BasisTime> SpaceSlice(
    double x, double y,
    const datastructures::DoubleTreeVector<
        BasisTime, space::HierarchicalBasisFn> &dbltree) {
  datastructures::TreeVector<BasisTime> space_slice(dbltree.root()->node_0());

  for (auto psi_space : dbltree.Project_1()->Bfs()) {
    double space_val = psi_space->node()->Eval(x, y);
    if (space_val != 0) {
      space_slice.root()->Union(
          psi_space->FrozenOtherAxis(),
          /* call_filter*/ datastructures::func_true, /* call_postprocess*/
          [space_val](const auto &my_node, const auto &other_node) {
            my_node->set_value(my_node->value() +
                               space_val * other_node->value());
          });
    }
  }
  return space_slice;
}

template <typename DblVecTree>
void TestNoEmptyFrozenAxes(std::shared_ptr<DblVecTree> vec) {
  for (auto phi : vec->Project_0()->Bfs())
    ASSERT_GT(phi->FrozenOtherAxis()->Bfs().size(), 0);
  for (auto phi : vec->Project_1()->Bfs())
    ASSERT_GT(phi->FrozenOtherAxis()->Bfs().size(), 0);
}

TEST(AdaptiveHeatEquation, CompareToPython) {
  for (bool use_cache : {true, false}) {
    auto B = Time::Bases();
    auto T = space::InitialTriangulation::UnitSquare();
    T.hierarch_basis_tree.UniformRefine(1);
    B.ortho_tree.UniformRefine(1);
    B.three_point_tree.UniformRefine(1);

    auto time_g1 = [](double t) { return -2 * (1 + t * t); };
    auto space_g1 = [](double x, double y) {
      return (x - 1) * x + (y - 1) * y;
    };
    auto time_g2 = [](double t) { return 2 * t; };
    auto space_g2 = [](double x, double y) {
      return (x - 1) * x * (y - 1) * y;
    };
    auto u0 = [](double x, double y) { return (1 - x) * x * (1 - y) * y; };

    auto [g_lf, u0_lf] = SmoothProblem();
    auto vec_Xd = std::make_shared<
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    vec_Xd->SparseRefine(1);

    AdaptiveHeatEquationOptions opts;
    opts.estimate_mean_zero = false;
    opts.use_cache = use_cache;
    opts.mark_theta = 0.7;
    opts.PX_alpha = 0.35;
    AdaptiveHeatEquation heat_eq(vec_Xd, std::move(g_lf), std::move(u0_lf),
                                 opts);

    std::vector<size_t> python_pcg_iters{2, 3, 5, 5, 5};
    auto [u, pcg_data] = heat_eq.Solve();
    vec_Xd->FromVectorContainer(u);
    auto u_nodes = vec_Xd->Bfs();
    Eigen::VectorXd python_u(u_nodes.size());
    python_u << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.056793956002164525, 0.12413158640093236;
    for (size_t i = 0; i < u_nodes.size(); i++)
      ASSERT_NEAR(u_nodes[i]->value(), python_u[i], 1e-5);
    ASSERT_NEAR(pcg_data.iterations, python_pcg_iters[0], 1);

    auto [residual, global_errors] = heat_eq.Estimate(u);
    auto [residual_norm, global_error] = global_errors;
    auto residual_nodes = residual->Bfs();
    Eigen::VectorXd python_residual(residual_nodes.size());
    python_residual << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 5.551115123125783e-17, 5.551115123125783e-17, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, -0.01628850213423607, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.004904405566708146,
        -0.004904405566708167, -0.004904405566708146, -0.004904405566708146,
        -0.006462116191632044, -0.006462116191632037, -0.006462116191632044,
        -0.006462116191632016;
    for (size_t i = 0; i < residual_nodes.size(); i++)
      ASSERT_NEAR(residual_nodes[i]->value(), python_residual[i], 1e-5);

    std::vector<std::pair<int, std::pair<double, double>>> python_mark_data{
        {{1, {0.022990516747664815, 0.020106102575435606}},
         {4, {0.016706324583205395, 0.015609386578623811}},
         {10, {0.08984241341963645, 0.07872609720998812}},
         {13, {0.07019458968270276, 0.059311900122128475}},
         {26, {0.050368099941736744, 0.04315894614947385}}}};

    auto marked_nodes = heat_eq.Mark(residual);
    ASSERT_EQ(marked_nodes.size(), python_mark_data[0].first);
    ASSERT_NEAR(residual_norm, python_mark_data[0].second.first, 1e-10);
    ASSERT_NEAR(global_error.error / python_mark_data[0].second.second, 1.0,
                0.2);

    vec_Xd->FromVectorContainer(u);
    heat_eq.Refine(marked_nodes);

    for (size_t iter = 1; iter < 5; iter++) {
      auto [u, pcg_data] =
          heat_eq.Solve(vec_Xd->ToVectorContainer(), heat_eq.RHS());
      auto [residual, global_errors] = heat_eq.Estimate(u);
      auto [residual_norm, global_error] = global_errors;
      auto marked_nodes = heat_eq.Mark(residual);
      ASSERT_EQ(marked_nodes.size(), python_mark_data[iter].first);
      ASSERT_NEAR(residual_norm, python_mark_data[iter].second.first, 1e-5);
      ASSERT_NEAR(global_error.error / python_mark_data[iter].second.second,
                  1.0, 0.1);
      ASSERT_NEAR(pcg_data.iterations, python_pcg_iters[iter], 1);
      TestNoEmptyFrozenAxes(heat_eq.vec_Xd());
      TestNoEmptyFrozenAxes(heat_eq.vec_Xdd());
      TestNoEmptyFrozenAxes(heat_eq.vec_Ydd());

      // Check that Ydd = GenerateTheta(Xdd, Ydd).
      auto theta =
          spacetime::GenerateTheta(*heat_eq.vec_Xdd(), *heat_eq.vec_Ydd());
      auto theta_nodes = theta->Bfs();
      auto ydd_nodes = heat_eq.vec_Ydd()->Bfs();
      ASSERT_EQ(theta_nodes.size(), ydd_nodes.size());
      ASSERT_EQ(theta->container().size(),
                heat_eq.vec_Ydd()->container().size());
      for (int i = 0; i < theta_nodes.size(); i++)
        ASSERT_EQ(theta_nodes[i]->nodes(), ydd_nodes[i]->nodes());

      vec_Xd->FromVectorContainer(u);
      heat_eq.Refine(marked_nodes);
    }
  }
}

TEST(AdaptiveHeatEquation, MovingPeak) {
  bool use_cache = false;
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(1);
  B.ortho_tree.UniformRefine(1);
  B.three_point_tree.UniformRefine(1);

  auto u = [](double t, double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y) *
           exp(-100 * ((t - x) * (t - x) + (t - y) * (t - y)));
  };
  auto vec_Xd = std::make_shared<
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
  vec_Xd->SparseRefine(1);
  auto [g_lf, u0_lf] = MovingPeakProblem(vec_Xd);

  AdaptiveHeatEquationOptions opts;
  AdaptiveHeatEquation heat_eq(vec_Xd, std::move(g_lf), std::move(u0_lf), opts);

  Eigen::VectorXd solution = Eigen::VectorXd::Zero(vec_Xd->container().size());
  std::vector<double> err_inf;
  std::vector<double> err_X;
  std::vector<double> size_X;
  for (int i = 0; i < 9; i++) {
    size_X.push_back(vec_Xd->Bfs().size());

    // Solve.
    auto [new_solution, pcg_data] =
        heat_eq.Solve(solution, heat_eq.RHS(), 1e-8,
                      tools::linalg::StoppingCriterium::Relative);
    solution = new_solution;

    // Estimate.
    auto [residual, global_errors] = heat_eq.Estimate(solution);
    auto [residual_norm, global_error] = global_errors;
    err_X.emplace_back(global_error.error);
    if (i > 5) {
      double rate =
          log(err_X[i] / err_X[i - 1]) / log(size_X[i - 1] / size_X[i]);
      ASSERT_GE(rate, 0.5);
    }

    // Estimate the infinity norm.
    double err_inf_norm = 0.0;
    std::vector<double> pts{
        0.,         0.00961538, 0.01923077, 0.02884615, 0.03846154, 0.04807692,
        0.05769231, 0.06730769, 0.07692308, 0.08653846, 0.09615385, 0.10576923,
        0.11538462, 0.125,      0.13461538, 0.14423077, 0.15384615, 0.16346154,
        0.17307692, 0.18269231, 0.19230769, 0.20192308, 0.21153846, 0.22115385,
        0.23076923, 0.24038462, 0.25,       0.25961538, 0.26923077, 0.27884615,
        0.28846154, 0.29807692, 0.30769231, 0.31730769, 0.32692308, 0.33653846,
        0.34615385, 0.35576923, 0.36538462, 0.375,      0.38461538, 0.39423077,
        0.40384615, 0.41346154, 0.42307692, 0.43269231, 0.44230769, 0.45192308,
        0.46153846, 0.47115385, 0.48076923, 0.49038462, 0.5,        0.50961538,
        0.51923077, 0.52884615, 0.53846154, 0.54807692, 0.55769231, 0.56730769,
        0.57692308, 0.58653846, 0.59615385, 0.60576923, 0.61538462, 0.625,
        0.63461538, 0.64423077, 0.65384615, 0.66346154, 0.67307692, 0.68269231,
        0.69230769, 0.70192308, 0.71153846, 0.72115385, 0.73076923, 0.74038462,
        0.75,       0.75961538, 0.76923077, 0.77884615, 0.78846154, 0.79807692,
        0.80769231, 0.81730769, 0.82692308, 0.83653846, 0.84615385, 0.85576923,
        0.86538462, 0.875,      0.88461538, 0.89423077, 0.90384615, 0.91346154,
        0.92307692, 0.93269231, 0.94230769, 0.95192308, 0.96153846, 0.97115385,
        0.98076923, 0.99038462, 1.};
    vec_Xd->FromVectorContainer(solution);
    double max_t, max_x, max_y;
    for (double x : pts)
      for (double y : pts) {
        if (abs(x - y) > 0.15) continue;
        auto space_slice = SpaceSlice(x, y, *vec_Xd);
        auto space_slice_nodes = space_slice.Bfs();
        auto u_delta = [&](double t) {
          double result = 0;
          for (int i = 0; i < space_slice_nodes.size(); ++i) {
            result += space_slice_nodes[i]->node()->Eval(t) *
                      space_slice_nodes[i]->value();
          }
          return result;
        };

        for (double t : pts) {
          if (abs(x - t) > 0.15 || abs(y - t) > 0.15) continue;
          double error_txy = abs(u(t, x, y) - u_delta(t));
          if (error_txy > err_inf_norm) {
            err_inf_norm = error_txy;
            max_t = t;
            max_x = x;
            max_y = y;
          }
        }
      }
    err_inf.emplace_back(err_inf_norm);
    if (i > 5) {
      double rate =
          log(err_inf[i] / err_inf[i - 1]) / log(size_X[i - 1] / size_X[i]);
      std::cout << rate << std::endl;

      ASSERT_LE(err_inf[i], 7);
      ASSERT_LE(err_inf[i], err_inf[i - 1]);
      ASSERT_GE(rate, 0.5);
    }

    // Mark - refine
    auto marked_nodes = heat_eq.Mark(residual);
    vec_Xd->FromVectorContainer(solution);
    heat_eq.Refine(marked_nodes);
    solution = vec_Xd->ToVectorContainer();
  }
}
}  // namespace applications
