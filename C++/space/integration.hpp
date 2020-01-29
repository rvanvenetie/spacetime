#pragma once

#include <vector>

#include "triangulation.hpp"

namespace space {
template <unsigned O>
class IntegrationRule {
 public:
  template <class F>
  static double Integrate(F f, const Element2D &elem);

  // Vector of tuples <bary2, bary3, weight>.
  static std::vector<std::tuple<double, double, double>> rule;
};
};  // namespace space

#include "integration.ipp"
