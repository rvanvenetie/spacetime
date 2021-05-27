#include "linear_functional.hpp"

#include <functional>

#include "bases.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "integration.hpp"

namespace Time {
TEST(LinearFunctional, ZeroEvalFunctional) {
  Bases B;

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  B.three_point_tree.UniformRefine(ml);

  auto zero_eval_func = ZeroEvalFunctional<ContLinearScalingFn>();
  auto phis = B.cont_lin_tree.Bfs();
  for (auto phi : phis)
    ASSERT_NEAR(zero_eval_func.Eval(phi), phi->Eval(0.0), 1e-10);

  auto Delta = B.cont_lin_tree.NodesPerLevel();
  for (int l = 0; l < ml; l++) {
    auto vec =
        zero_eval_func.Eval(SparseIndices<ContLinearScalingFn>{Delta[l]});
    ASSERT_EQ(vec.size(), Delta[l].size());
    for (int i = 0; i < vec.size(); i++)
      ASSERT_NEAR(vec[i].second, Delta[l][i]->Eval(0.0), 1e-10);
  }
}

TEST(LinearFunctional, QuadratureFunctional) {
  Bases B;

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  B.three_point_tree.UniformRefine(ml);

  std::function<double(double)> f([](double t) { return t * t; });
  auto quad_func = QuadratureFunctional<ContLinearScalingFn>(f, /*order*/ 2);
  auto phis = B.cont_lin_tree.Bfs();
  for (auto phi : phis) {
    double ip = 0.0;
    for (auto elem : phi->support())
      ip +=
          Integrate([phi, &f](double t) { return phi->Eval(t) * f(t); }, *elem,
                    /*degree*/ 3);
    ASSERT_NEAR(quad_func.Eval(phi), ip, 1e-10);
  }

  auto Delta = B.cont_lin_tree.NodesPerLevel();
  for (int l = 0; l < ml; l++) {
    auto vec = quad_func.Eval(SparseIndices<ContLinearScalingFn>{Delta[l]});
    ASSERT_EQ(vec.size(), Delta[l].size());
    for (int i = 0; i < vec.size(); i++) {
      auto phi = Delta[l][i];
      double ip = 0.0;
      for (auto elem : phi->support())
        ip += Integrate([phi, &f](double t) { return phi->Eval(t) * f(t); },
                        *elem,
                        /*degree*/ 3);
      ASSERT_NEAR(vec[i].second, ip, 1e-10);
    }
  }
}

TEST(LinearFunctional, AverageFunctional) {
  Bases B;

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  B.three_point_tree.UniformRefine(ml);

  std::function<double(double)> f([](double t) { return 1; });
  auto quad_func = QuadratureFunctional<ContLinearScalingFn>(f, /*order*/ 0);
  auto phis = B.cont_lin_tree.Bfs();
  for (auto phi : phis) {
    double ip = 0.0;
    for (auto elem : phi->support())
      ip += Integrate([phi, &f](double t) { return phi->Eval(t); }, *elem,
                      /*degree*/ 1);
    ASSERT_NEAR(quad_func.Eval(phi), ip, 1e-10);
  }
}
};  // namespace Time
