#include "linear_form.hpp"

#include "integration.hpp"

namespace space {
namespace {
double EvalHatFn(double x, double y, Element2D *elem, size_t i) {
  auto bary = elem->BarycentricCoordinates(x, y);
  assert((bary.array() >= 0).all());
  return bary[i];
}
}  // namespace

std::array<double, 3> LinearForm::QuadEval(Element2D *elem) const {
  std::array<double, 3> result;
  for (size_t i = 0; i < 3; i++)
    result[i] = Integrate(
        [&](double x, double y) { return f_(x, y) * EvalHatFn(x, y, elem, i); },
        *elem, quad_order_ + 1);
  return result;
}
}  // namespace space
