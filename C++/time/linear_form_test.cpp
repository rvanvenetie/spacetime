#include "linear_form.hpp"

#include "bases.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "integration.hpp"

using datastructures::TreeVector;

namespace Time {
TEST(LinearForm, ThreePointQuadratureTest) {
  Bases B;

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  B.three_point_tree.UniformRefine(ml);

  auto f = [](double t) { return t * t; };
  auto linear_form = LinearForm<ThreePointWaveletFn>(
      std::make_unique<QuadratureFunctional<ContLinearScalingFn>>(f,
                                                                  /*order*/ 2));
  auto vec_out =
      TreeVector<ThreePointWaveletFn>(B.three_point_tree.meta_root());
  vec_out.DeepRefine();

  linear_form.Apply(vec_out.root());

  for (auto nv : vec_out.Bfs()) {
    auto phi = nv->node();
    double ip = 0.0;
    for (auto elem : phi->support())
      ip +=
          Integrate([phi, &f](double t) { return phi->Eval(t) * f(t); }, *elem,
                    /*degree*/ 3);
    ASSERT_NE(nv->value(), 0.0);
    ASSERT_NEAR(nv->value(), ip, 1e-10);
  }
}

TEST(LinearForm, ThreePointZeroEvalTest) {
  Bases B;

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  B.three_point_tree.UniformRefine(ml);

  auto linear_form = LinearForm<ThreePointWaveletFn>(
      std::make_unique<ZeroEvalFunctional<ContLinearScalingFn>>());
  auto vec_out =
      TreeVector<ThreePointWaveletFn>(B.three_point_tree.meta_root());
  vec_out.DeepRefine();

  linear_form.Apply(vec_out.root());

  for (auto nv : vec_out.Bfs())
    ASSERT_NEAR(nv->value(), nv->node()->Eval(0.0), 1e-10);
}

TEST(LinearForm, OrthoQuadratureTest) {
  Bases B;

  int ml = 6;
  // Now we check what happens when we also refine near the end points.
  B.ortho_tree.UniformRefine(ml);

  auto f = [](double t) { return t * t * t; };
  auto linear_form = LinearForm<OrthonormalWaveletFn>(
      std::make_unique<QuadratureFunctional<DiscLinearScalingFn>>(f,
                                                                  /*order*/ 3));
  auto vec_out = TreeVector<OrthonormalWaveletFn>(B.ortho_tree.meta_root());
  vec_out.DeepRefine();

  linear_form.Apply(vec_out.root());

  for (auto nv : vec_out.Bfs()) {
    auto phi = nv->node();
    double ip = 0.0;
    for (auto elem : phi->support())
      ip +=
          Integrate([phi, &f](double t) { return phi->Eval(t) * f(t); }, *elem,
                    /*degree*/ 4);
    ASSERT_NE(nv->value(), 0.0);
    ASSERT_NEAR(nv->value(), ip, 1e-10);
  }
}
};  // namespace Time
