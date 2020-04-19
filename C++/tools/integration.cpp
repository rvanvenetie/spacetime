#include "integration.hpp"

namespace {
#include "integration_rules.ipp"
}

namespace tools {
double Integrate1D(const std::function<double(double)>& f,
                   const Time::Element1D& elem, size_t degree) {
  assert(degree <= integration_rules_1d.size());
  double integral = 0.0;
  for (const auto [p, w] : integration_rules_1d[degree]) {
    auto x = elem.GlobalCoordinates(p);
    integral += w * f(x);
  }
  return elem.area() * integral;
}

double Integrate2D(const std::function<double(double, double)>& f,
                   const space::Element2D& elem, size_t degree) {
  assert(degree <= integration_rules_2d.size());
  double integral = 0.0;
  for (const auto [p, q, w] : integration_rules_2d[degree]) {
    auto [x, y] = elem.GlobalCoordinates(p, q);
    integral += w * f(x, y);
  }
  return elem.area() * integral;
}
}  // namespace tools
