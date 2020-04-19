#include "integration.hpp"

namespace {
#include "integration_rules.ipp"
}

namespace space {
double Integrate(const std::function<double(double, double)>& f,
                 const Element2D& elem, size_t degree) {
  assert(degree <= integration_rules.size());
  double integral = 0.0;
  for (const auto [p, q, w] : integration_rules[degree]) {
    auto [x, y] = elem.GlobalCoordinates(p, q);
    integral += w * f(x, y);
  }
  return elem.area() * integral;
}
}  // namespace space
