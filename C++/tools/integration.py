""" Usage: python3 integration.py """
import quadpy

#(dim, dir)
files = [(1, 'time'), (2, 'space')]

max_degree = 10
for dim, dirname in files:
    with open('../%s/integration_rules.ipp' % dirname, 'w') as fn:
        print(file=fn)
        print("std::array<std::vector<std::array<double, %d>>, %d> " +
              "integration_rules{{" % (dim + 1, max_degree + 1),
              file=fn)
        for degree in range(max_degree + 1):
            rule = quadpy.nsimplex.grundmann_moeller(n=dim,
                                                     s=max(0, degree // 2))
            npoints = len(rule.weights)
            print("{", file=fn)
            for p in range(npoints):
                print("\t\t{", end='', file=fn)
                for pt in rule.points[p][1:]:
                    print("%.20f, " % pt, end='', file=fn)
                print("%.20f}," % rule.weights[p], file=fn)
            print("},", file=fn)
        print("}};", file=fn)
