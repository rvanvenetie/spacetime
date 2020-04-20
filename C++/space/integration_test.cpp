#include "integration.hpp"

#include <cmath>
#include "../datastructures/multi_tree_view.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "initial_triangulation.hpp"

using datastructures::static_for;
<<<<<<< HEAD:C++/space/integration_test.cpp
namespace space {
TEST(Integration, ProductOfMonomials) {
=======
using space::InitialTriangulation;
using Time::elem_tree;

namespace tools {
TEST(Integration1D, Monomials) {
  elem_tree.UniformRefine(4);

  static_for<10>([&](auto degree) {
    for (auto elem : elem_tree.Bfs()) {
      for (size_t n = 0; n < degree; n++) {
        auto f = [n](double x) { return pow(x, n); };
        auto [a, b] = elem->Interval();
        auto result = IntegrationRule<1, degree>::Integrate(f, *elem);
        auto expected = (pow(b, n + 1) - pow(a, n + 1)) / (n + 1);
        EXPECT_NEAR(result, expected, 1e-10);
      }
    }
  });
}

TEST(Integration2D, ProductOfMonomials) {
>>>>>>> parent of 3d7665c... bla:C++/tools/integration_test.cpp
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
