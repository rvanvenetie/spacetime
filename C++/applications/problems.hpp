#pragma once
#include <cmath>
#include "../spacetime/linear_form.hpp"
#include "../time/basis.hpp"

namespace applications {

using spacetime::CreateQuadratureTensorLinearForm;
using spacetime::CreateZeroEvalLinearForm;
using spacetime::NoOpLinearForm;
using spacetime::SumTensorLinearForm;

// Solution u = (1 + t^2) x (1-x) y (1-y) on unit square.
std::pair<std::unique_ptr<LinearFormBase<Time::OrthonormalWaveletFn>>,
          std::unique_ptr<LinearFormBase<Time::ThreePointWaveletFn>>>
SmoothProblem() {
  auto time_g1 = [](double t) { return -2 * (1 + t * t); };
  auto space_g1 = [](double x, double y) { return (x - 1) * x + (y - 1) * y; };
  auto time_g2 = [](double t) { return 2 * t; };
  auto space_g2 = [](double x, double y) { return (x - 1) * x * (y - 1) * y; };
  auto u0 = [](double x, double y) { return (1 - x) * x * (1 - y) * y; };

  return {std::make_unique<SumTensorLinearForm<Time::OrthonormalWaveletFn>>(
              CreateQuadratureTensorLinearForm<Time::OrthonormalWaveletFn>(
                  time_g1, space_g1, 2, 2),
              CreateQuadratureTensorLinearForm<Time::OrthonormalWaveletFn>(
                  time_g2, space_g2, 1, 4)),
          CreateZeroEvalLinearForm<Time::ThreePointWaveletFn>(
              u0, /* apply_quadrature*/ true, /* quadrature_order*/ 4)};
}

// Singular problem: u0 = 1, f = 0.
std::pair<std::unique_ptr<LinearFormBase<Time::OrthonormalWaveletFn>>,
          std::unique_ptr<LinearFormBase<Time::ThreePointWaveletFn>>>
SingularProblem() {
  auto u0 = [](double x, double y) { return 1.0; };

  return {std::make_unique<NoOpLinearForm<Time::OrthonormalWaveletFn>>(),
          CreateZeroEvalLinearForm<Time::ThreePointWaveletFn>(
              u0, /* apply_quadrature*/ true, /* quadrature_order */ 1)};
}

// Problem with u0 = 0 and rhs f = t * 1_{x^2 + y^2 < 1/4}.
std::pair<std::unique_ptr<LinearFormBase<Time::OrthonormalWaveletFn>>,
          std::unique_ptr<LinearFormBase<Time::ThreePointWaveletFn>>>
CylinderProblem(size_t space_order = 2) {
  auto time_f = [](double t) { return t; };
  auto space_f = [](double x, double y) { return (x * x + y * y < 0.25); };
  return {CreateQuadratureTensorLinearForm<Time::OrthonormalWaveletFn>(
              time_f, space_f, /* time_order */ 1, space_order),
          std::make_unique<NoOpLinearForm<Time::ThreePointWaveletFn>>()};
}

// Problem with u = sin(pi x) sin(pi y) exp(-100 ((t-x)^2 + (t-y)^2))
std::pair<std::unique_ptr<LinearFormBase<Time::OrthonormalWaveletFn>>,
          std::unique_ptr<LinearFormBase<Time::ThreePointWaveletFn>>>
MovingPeakProblem(size_t space_order = 2) {
  auto u0 = [](double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y) * exp(-100 * (x * x + y * y));
  };
  auto g = [](double x, double y, double t) {
    double dt_u = -100 * ((x - t) * (x - t) + (y - t) * (y - t)) *
                  sin(M_PI * x) * sin(M_PI * y) *
                  exp(-100 * ((x - t) * (x - t) + (y - t) * (y - t)));
    double dxx_u = sin(M_PI * y) *
                   (400 * M_PI * (t - x) * cos(M_PI * x) +
                    200 * (200 * (t - x) * (t - x) - 1) * sin(M_PI * x) -
                    M_PI * M_PI * sin(M_PI * x)) *
                   exp(-100 * ((x - t) * (x - t) + (y - t) * (y - t)));
    double dyy_u = sin(M_PI * x) *
                   (400 * M_PI * (t - y) * cos(M_PI * y) +
                    200 * (200 * (t - y) * (t - y) - 1) * sin(M_PI * y) -
                    M_PI * M_PI * sin(M_PI * y)) *
                   exp(-100 * ((x - t) * (x - t) + (y - t) * (y - t)));
    return dt_u - dxx_u - dxx_u;
  };
  return {CreateQuadratureLinearForm<Time::OrthonormalWaveletFn>(
              time_f, space_f, /* time_order */ 1, space_order),
          CreateZeroEvalLinearForm<Time::ThreePointWaveletFn>(u0, 1)};
}

}  // namespace applications
