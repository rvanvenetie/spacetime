#include "integration.hpp"

#include <cmath>

#include "../datastructures/multi_tree_view.hpp"
#include "basis.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::static_for;

namespace Time {
// TEST(Integration, Monomials) {
//   auto &elem_tree = ElementTree();
//   elem_tree.UniformRefine(4);
// 
//   static_for<10>([&](auto degree) {
//     for (auto elem : elem_tree.Bfs()) {
//       for (size_t n = 0; n < degree; n++) {
//         auto f = [n](double x) { return pow(x, n); };
//         auto [a, b] = elem->Interval();
//         auto result = Integrate(f, *elem, degree);
//         auto expected = (pow(b, n + 1) - pow(a, n + 1)) / (n + 1);
//         EXPECT_NEAR(result, expected, 1e-10);
//       }
//     }
//   });
// }
};  // namespace Time
