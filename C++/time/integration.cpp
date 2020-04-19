#include "integration.hpp"

namespace {
#include "integration_rules.ipp"
}
namespace Time {
double Integrate(const std::function<double(double)>& f, const Element1D& elem,
                 size_t degree) {
  assert(degree <= integration_rules.size());
  double integral = 0.0;
  for (const auto [p, w] : integration_rules[degree]) {
    auto x = elem.GlobalCoordinates(p);
    integral += w * f(x);
  }
  return elem.area() * integral;
}
}  // namespace Time
