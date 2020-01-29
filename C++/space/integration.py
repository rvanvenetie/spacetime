""" Usage: python3 integration.py > integration.ipp """
import quadpy

print("""#pragma once

#include "integration.hpp"

namespace space {
template <unsigned P>
template <class F>
double IntegrationRule<P>::Integrate(F f, const Element2D &elem) {
  double integral = 0.0;
  for (const auto [p, q, w] : rule) {
    auto [x, y] = elem.GlobalCoordinates(p, q);
    integral += w * f(x, y);
  }
  return elem.area() * integral;
}
""")
for order in range(10):
    nco = quadpy.triangle.newton_cotes_open(order)
    npoints = len(nco.weights)
    print("template <>")
    print("std::vector<std::tuple<double, double, double>>"
          " IntegrationRule<%d>::rule{" % order)
    for p in range(npoints):
        print(
            "\t\t{%.20f, %.20f, %.20f}," %
            (nco.points[p][1], nco.points[p][2], nco.weights[p]), )
    print("};")
print("};  // namespace space")
