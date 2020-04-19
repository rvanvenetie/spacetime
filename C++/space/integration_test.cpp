#include "integration.hpp"

#include <cmath>

#include "../datastructures/multi_tree_view.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "initial_triangulation.hpp"

using datastructures::static_for;

namespace space {
TEST(Integration, ProductOfMonomials) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(4);

  static_for<10>([&T](auto degree) {
    for (auto elem : T.elem_tree.Bfs()) {
      auto f = [](double x, double y) { return 1.0; };
      auto result = Integrate(f, *elem, degree);
      EXPECT_NEAR(result, elem->area(), 1e-10);
    }
  });

  static_for<10>([&T](auto degree) {
    auto root = T.elem_tree.Bfs()[0];
    for (size_t n = 0; n < degree; n++) {
      for (size_t m = 0; m + n <= degree; m++) {
        auto f = [n, m](double x, double y) { return pow(x, n) * pow(y, m); };
        auto result = Integrate(f, *root, degree);
        auto expected =
            tgamma(2 + m) * tgamma(1 + n) / ((1 + m) * tgamma(3 + m + n));
        EXPECT_NEAR(result, expected, 1e-10);
      }
    }
  });
}
};  // namespace space
