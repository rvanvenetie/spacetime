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
MovingPeakProblem(std::shared_ptr<datastructures::DoubleTreeVector<
                      Time::ThreePointWaveletFn, space::HierarchicalBasisFn>>
                      X_delta) {
  auto u0 = [](double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y) * exp(-100 * (x * x + y * y));
  };
  auto g = [](double t, double x, double y) {
    double sinPIx = sin(M_PI * x);
    double sinPIy = sin(M_PI * y);
    double cosPIx = cos(M_PI * x);
    double cosPIy = cos(M_PI * y);
    double exptxy = exp(-100 * ((x - t) * (x - t) + (y - t) * (y - t)));

    double dt_u =
        -100 * (-2 * (x - t) - 2 * (y - t)) * sinPIx * sinPIy * exptxy;
    double dxx_u =
        sinPIy *
        (400 * M_PI * (t - x) * cosPIx +
         200 * (200 * (t - x) * (t - x) - 1) * sinPIx - M_PI * M_PI * sinPIx) *
        exptxy;
    double dyy_u =
        sinPIx *
        (400 * M_PI * (t - y) * cosPIy +
         200 * (200 * (t - y) * (t - y) - 1) * sinPIy - M_PI * M_PI * sinPIy) *
        exptxy;
    return dt_u - dxx_u - dxx_u;
  };
  return {std::make_unique<spacetime::InterpolationLinearForm>(X_delta, g),
          CreateZeroEvalLinearForm<Time::ThreePointWaveletFn>(
              u0, /* apply_quadrature*/ false)};
}

}  // namespace applications
