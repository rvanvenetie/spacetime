""" Usage: python3 integration.py > integration_rules.ipp """
import quadpy

max_degree = 10
for dim in [1, 2]:
    print()
    print(
        "std::array<std::vector<std::array<double, %d>>, %d> integration_rules_%dd{{"
        % (dim + 1, max_degree + 1, dim))
    for degree in range(max_degree + 1):
        rule = quadpy.nsimplex.grundmann_moeller(n=dim, s=max(0, degree // 2))
        npoints = len(rule.weights)
        print("{")
        for p in range(npoints):
            print("\t\t{", end='')
            for pt in rule.points[p][1:]:
                print("%.20f, " % pt, end='')
            print("%.20f}," % rule.weights[p])
        print("},")
    print("}};")
