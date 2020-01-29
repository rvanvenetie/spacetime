import quadpy

print("namespace space {")
for order in range(10):
    nco = quadpy.triangle.newton_cotes_open(order)
    npoints = len(nco.weights)
    print("template <>")
    print("std::vector<std::tuple<double, double, double>>"
          " IntegrationRule<%d>::rule{" % order,
          end='')
    for p in range(npoints):
        print("\n\t{%.20f,%.20f,%.20f, %d}," %
              (nco.points[p][1], nco.points[p][2], nco.weights[p], nco.degree),
              end='')
    print("};")
print("};  // namespace space")
