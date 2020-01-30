#pragma once

#include <vector>

#include "../space/triangulation.hpp"
#include "../time/basis.hpp"

namespace tools {
template <unsigned dim, unsigned degree>
class IntegrationRule {
 public:
  template <class Function, class Element>
  static double Integrate(const Function& f, const Element& elem);

  // Vector of tuples <bary2, ..., bary_{dim+1}, weight>.
  static std::vector<std::array<double, dim + 1>> rule;
};

template <unsigned degree>
class IntegrationRule<1, degree> {
 public:
  template <class Function>
  static double Integrate(const Function& f, const Time::Element1D& elem) {
    double integral = 0.0;
    for (const auto [p, w] : rule) {
      auto x = elem.GlobalCoordinates(p);
      integral += w * f(x);
    }
    return elem.area() * integral;
  }

  static std::vector<std::array<double, 2>> rule;
};

template <unsigned degree>
class IntegrationRule<2, degree> {
 public:
  template <class Function>
  static double Integrate(const Function& f, const space::Element2D& elem) {
    double integral = 0.0;
    for (const auto [p, q, w] : rule) {
      auto [x, y] = elem.GlobalCoordinates(p, q);
      integral += w * f(x, y);
    }
    return elem.area() * integral;
  }

  static std::vector<std::array<double, 3>> rule;
};
};  // namespace tools

#include "integration.ipp"
