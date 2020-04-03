#include "../space/initial_triangulation.hpp"
#include "../spacetime/linear_form.hpp"
#include "../time/basis.hpp"

using spacetime::CreateQuadratureLinearForm;
using spacetime::CreateSumLinearForm;
using spacetime::CreateZeroEvalLinearForm;

// Solution u = (1 + t^2) x (1-x) y (1-y) on unit square.
auto SmoothProblem() {
  std::function<double(double)> time_g1(
      [](double t) { return -2 * (1 + t * t); });
  std::function<double(double, double)> space_g1(
      [](double x, double y) { return (x - 1) * x + (y - 1) * y; });
  std::function<double(double)> time_g2([](double t) { return 2 * t; });
  std::function<double(double, double)> space_g2(
      [](double x, double y) { return (x - 1) * x * (y - 1) * y; });
  std::function<double(double, double)> u0(
      [](double x, double y) { return (1 - x) * x * (1 - y) * y; });

  return std::make_pair(
      CreateSumLinearForm<Time::OrthonormalWaveletFn>(
          CreateQuadratureLinearForm<Time::OrthonormalWaveletFn, 2, 2>(
              time_g1, space_g1),
          CreateQuadratureLinearForm<Time::OrthonormalWaveletFn, 1, 4>(
              time_g2, space_g2)),
      CreateZeroEvalLinearForm<Time::ThreePointWaveletFn, 4>(u0));
}

// Singular problem on unit square.
auto SingularProblem() {
  std::function<double(double)> time_g([](double t) { return 0; });
  std::function<double(double, double)> space_g(
      [](double x, double y) { return 0; });
  std::function<double(double, double)> u0(
      [](double x, double y) { return 1.0; });

  return std::make_pair(
      CreateQuadratureLinearForm<Time::OrthonormalWaveletFn, 0, 0>(time_g,
                                                                   space_g),
      CreateZeroEvalLinearForm<Time::ThreePointWaveletFn, 1>(u0));
}
