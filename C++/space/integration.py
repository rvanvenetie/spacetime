""" Usage: python3 integration.py > integration.ipp """
import quadpy

print("""#pragma once

#include "integration.hpp"

namespace space {""")
for degree in range(15):
    rule = quadpy.nsimplex.grundmann_moeller(n=2, s=max(0, degree // 2))
    npoints = len(rule.weights)
    print("template <>")
    print("std::vector<std::tuple<double, double, double>>"
          " IntegrationRule<%d>::rule{" % degree)
    for p in range(npoints):
        print(
            "\t\t{%.20f, %.20f, %.20f}," %
            (rule.points[p][1], rule.points[p][2], rule.weights[p]), )
    print("};")
print("};  // namespace space")
