#include "integration.hpp"

#include <cmath>
#include "../datastructures/multi_tree_view.hpp"
#include "../space/initial_triangulation.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::static_for;

namespace tools {
TEST(Integration1D, ProductOfMonomials) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(4);

  static_for<10>([&T](auto order) {
    for (auto elem : T.elem_tree.Bfs()) {
      auto f = [](double x, double y) { return 1.0; };
      EXPECT_NEAR(IntegrationRule<order>::Integrate(f, *elem), elem->area(),
                  1e-10);
    }
  });

  static_for<10>([&T](auto degree) {
    auto root = T.elem_tree.Bfs()[0];
    for (size_t n = 0; n < degree; n++) {
      for (size_t m = 0; m + n <= degree; m++) {
        auto f = [n, m](double x, double y) { return pow(x, n) * pow(y, m); };
        EXPECT_NEAR(
            IntegrationRule<degree>::Integrate(f, *root),
            tgamma(2 + m) * tgamma(1 + n) / ((1 + m) * tgamma(3 + m + n)),
            1e-10);
      }
    }
  });
}

TEST(Integration2D, ProductOfMonomials) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(4);

  static_for<10>([&T](auto order) {
    for (auto elem : T.elem_tree.Bfs()) {
      auto f = [](double x, double y) { return 1.0; };
      EXPECT_NEAR(IntegrationRule<order>::Integrate(f, *elem), elem->area(),
                  1e-10);
    }
  });

  static_for<10>([&T](auto degree) {
    auto root = T.elem_tree.Bfs()[0];
    for (size_t n = 0; n < degree; n++) {
      for (size_t m = 0; m + n <= degree; m++) {
        auto f = [n, m](double x, double y) { return pow(x, n) * pow(y, m); };
        EXPECT_NEAR(
            IntegrationRule<degree>::Integrate(f, *root),
            tgamma(2 + m) * tgamma(1 + n) / ((1 + m) * tgamma(3 + m + n)),
            1e-10);
      }
    }
  });
}
};  // namespace tools
