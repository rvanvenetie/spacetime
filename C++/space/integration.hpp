#pragma once

#include <vector>

#include "triangulation.hpp"

namespace space {
template <unsigned degree>
class IntegrationRule {
 public:
  template <class F>
  static double Integrate(const F& f, const Element2D& elem) {
    double integral = 0.0;
    for (const auto [p, q, w] : rule) {
      auto [x, y] = elem.GlobalCoordinates(p, q);
      integral += w * f(x, y);
    }
    return elem.area() * integral;
  }

  // Vector of tuples <bary2, bary3, weight>.
  static std::vector<std::tuple<double, double, double>> rule;
};
};  // namespace space

#include "integration.ipp"
