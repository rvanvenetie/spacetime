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
      auto [u, pcg_data] = heat_eq.Solve(vec_Xd->ToVectorContainer());
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
  for (int i = 0; i < 15; i++) {
    std::cout << "\n\niteration " << i;
    // Solve.
    auto [new_solution, pcg_data] = heat_eq.Solve(
        solution, 1e-6, tools::linalg::StoppingCriterium::Relative);
    solution = new_solution;

    // Estimate.
    auto [residual, global_errors] = heat_eq.Estimate(solution);
    auto [residual_norm, global_error] = global_errors;
    std::cout << "\n\tresidual-norm: " << residual_norm
              << "\n\tglobal-error: " << global_error.error
              << "\n\tYnorm-error: " << global_error.error_Yprime
              << "\n\tT0-error: " << global_error.error_t0 << std::flush;

    std::cout << "\n\ttrace-error-Xd: ";
    for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
      auto vec_Xd = heat_eq.vec_Xd();
      vec_Xd->FromVectorContainer(solution);
      double error_t = ErrorEstimator::ComputeTraceError(
          t, [&](double x, double y) { return u(t, x, y); }, vec_Xd.get());
      std::cout << "\n\t\t" << t << " : " << sqrt(error_t);
    }

    // Mark - refine
    auto marked_nodes = heat_eq.Mark(residual);
    vec_Xd->FromVectorContainer(solution);
    heat_eq.Refine(marked_nodes);
    solution = vec_Xd->ToVectorContainer();
  }
}
}  // namespace applications
