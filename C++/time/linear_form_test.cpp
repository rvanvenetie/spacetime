#include "linear_form.hpp"

#include "../tools/integration.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::TreeVector;

namespace Time {
TEST(LinearForm, ThreePointQuadratureTest) {
  ResetTrees();

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  three_point_tree.UniformRefine(ml);

  auto f = [](double t) { return t * t; };
  auto linear_form = LinearForm<ThreePointWaveletFn>(
      std::make_unique<QuadratureFunctional<ContLinearScalingFn>>(f,
                                                                  /*order*/ 2));
  auto vec_out = TreeVector<ThreePointWaveletFn>(three_point_tree.meta_root);
  vec_out.DeepRefine();

  linear_form.Apply(vec_out.root());

  for (auto nv : vec_out.Bfs()) {
    auto phi = nv->node();
    double ip = 0.0;
    for (auto elem : phi->support())
      ip += tools::Integrate1D(
          [phi, &f](double t) { return phi->Eval(t) * f(t); }, *elem, 3);
    ASSERT_NE(nv->value(), 0.0);
    ASSERT_NEAR(nv->value(), ip, 1e-10);
  }
}

TEST(LinearForm, ThreePointZeroEvalTest) {
  ResetTrees();

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  three_point_tree.UniformRefine(ml);

  auto linear_form = LinearForm<ThreePointWaveletFn>(
      std::make_unique<ZeroEvalFunctional<ContLinearScalingFn>>());
  auto vec_out = TreeVector<ThreePointWaveletFn>(three_point_tree.meta_root);
  vec_out.DeepRefine();

  linear_form.Apply(vec_out.root());

  for (auto nv : vec_out.Bfs())
    ASSERT_NEAR(nv->value(), nv->node()->Eval(0.0), 1e-10);
}

TEST(LinearForm, OrthoQuadratureTest) {
  ResetTrees();

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  ortho_tree.UniformRefine(ml);

  auto f = [](double t) { return t * t * t; };
  auto linear_form = LinearForm<OrthonormalWaveletFn>(
      std::make_unique<QuadratureFunctional<DiscLinearScalingFn>>(f,
                                                                  /*order*/ 2));
  auto vec_out = TreeVector<OrthonormalWaveletFn>(ortho_tree.meta_root);
  vec_out.DeepRefine();

  linear_form.Apply(vec_out.root());

  for (auto nv : vec_out.Bfs()) {
    auto phi = nv->node();
    double ip = 0.0;
    for (auto elem : phi->support())
      ip += tools::Integrate1D(
          [phi, &f](double t) { return phi->Eval(t) * f(t); }, *elem, 4);
    ASSERT_NE(nv->value(), 0.0);
    ASSERT_NEAR(nv->value(), ip, 1e-10);
  }
}
};  // namespace Time
