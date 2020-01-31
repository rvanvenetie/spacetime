""" Usage: python3 integration.py > integration.ipp """
import quadpy

print("""#pragma once

#include "integration.hpp"

namespace tools {""")
for dim in [1, 2]:
    print()
    for degree in range(11):
        rule = quadpy.nsimplex.grundmann_moeller(n=dim, s=max(0, degree // 2))
        npoints = len(rule.weights)
        print("template <>")
        print("const std::vector<std::array<double, %d>>"
              " IntegrationRule<%d, %d>::rule{" % (dim + 1, dim, degree))
        for p in range(npoints):
            print("\t\t{", end='')
            for pt in rule.points[p][1:]:
                print("%.20f, " % pt, end='')
            print("%.20f}," % rule.weights[p])
        print("};")
print("};  // namespace tools")
