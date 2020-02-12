#include "linear_functional.hpp"
#include "../tools/integration.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "three_point_basis.hpp"

namespace Time {
TEST(LinearFunctional, ZeroEvalFunctional) {
  ResetTrees();

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  three_point_tree.UniformRefine(ml);

  auto zero_eval_func = ZeroEvalFunctional<ContLinearScalingFn>();
  auto phis = cont_lin_tree.Bfs();
  for (auto phi : phis)
    ASSERT_NEAR(zero_eval_func.Eval(phi), phi->Eval(0.0), 1e-10);

  auto Delta = cont_lin_tree.NodesPerLevel();
  for (int l = 0; l < ml; l++) {
    auto vec =
        zero_eval_func.Eval(SparseIndices<ContLinearScalingFn>{Delta[l]});
    ASSERT_EQ(vec.size(), Delta[l].size());
    for (int i = 0; i < vec.size(); i++)
      ASSERT_NEAR(vec[i].second, Delta[l][i]->Eval(0.0), 1e-10);
  }
}

TEST(LinearFunctional, QuadratureFunctional) {
  ResetTrees();

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  three_point_tree.UniformRefine(ml);

  auto f = [](double t) { return t * t; };
  auto quad_func =
      QuadratureFunctional<ContLinearScalingFn, /*order*/ 2, decltype(f)>(f);
  auto phis = cont_lin_tree.Bfs();
  for (auto phi : phis) {
    double ip = 0.0;
    for (auto elem : phi->support())
      ip += tools::IntegrationRule</*dim*/ 1, /*degree*/ 3>::Integrate(
          [phi, &f](double t) { return phi->Eval(t) * f(t); }, *elem);
    ASSERT_NEAR(quad_func.Eval(phi), ip, 1e-10);
  }

  auto Delta = cont_lin_tree.NodesPerLevel();
  for (int l = 0; l < ml; l++) {
    auto vec = quad_func.Eval(SparseIndices<ContLinearScalingFn>{Delta[l]});
    ASSERT_EQ(vec.size(), Delta[l].size());
    for (int i = 0; i < vec.size(); i++) {
      auto phi = Delta[l][i];
      double ip = 0.0;
      for (auto elem : phi->support())
        ip += tools::IntegrationRule</*dim*/ 1, /*degree*/ 3>::Integrate(
            [phi, &f](double t) { return phi->Eval(t) * f(t); }, *elem);
      ASSERT_NEAR(vec[i].second, ip, 1e-10);
    }
  }
}
};  // namespace Time
