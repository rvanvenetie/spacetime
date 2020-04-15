#pragma once
#include "../spacetime/linear_form.hpp"
#include "../time/basis.hpp"

namespace applications {

using spacetime::CreateQuadratureLinearForm;
using spacetime::CreateZeroEvalLinearForm;
using spacetime::SumLinearForm;

// Solution u = (1 + t^2) x (1-x) y (1-y) on unit square.
std::pair<std::unique_ptr<LinearFormBase<Time::OrthonormalWaveletFn>>,
          std::unique_ptr<LinearFormBase<Time::ThreePointWaveletFn>>>
SmoothProblem() {
  auto time_g1 = [](double t) { return -2 * (1 + t * t); };
  auto space_g1 = [](double x, double y) { return (x - 1) * x + (y - 1) * y; };
  auto time_g2 = [](double t) { return 2 * t; };
  auto space_g2 = [](double x, double y) { return (x - 1) * x * (y - 1) * y; };
  auto u0 = [](double x, double y) { return (1 - x) * x * (1 - y) * y; };

  return {std::make_unique<SumLinearForm<Time::OrthonormalWaveletFn>>(
              CreateQuadratureLinearForm<Time::OrthonormalWaveletFn, 2, 2>(
                  time_g1, space_g1),
              CreateQuadratureLinearForm<Time::OrthonormalWaveletFn, 1, 4>(
                  time_g2, space_g2)),
          CreateZeroEvalLinearForm<Time::ThreePointWaveletFn, 4>(u0)};
}

// Singular problem on unit square.
std::pair<std::unique_ptr<LinearFormBase<Time::OrthonormalWaveletFn>>,
          std::unique_ptr<LinearFormBase<Time::ThreePointWaveletFn>>>
SingularProblem() {
  auto time_g = [](double t) { return 0; };
  auto space_g = [](double x, double y) { return 0; };
  auto u0 = [](double x, double y) { return 1.0; };

  return {CreateQuadratureLinearForm<Time::OrthonormalWaveletFn, 0, 0>(time_g,
                                                                       space_g),
          CreateZeroEvalLinearForm<Time::ThreePointWaveletFn, 1>(u0)};
}
}  // namespace applications
