#pragma once
#include <stddef.h>

#include <utility>

#include "triangulation.hpp"

namespace space {

class LinearForm {
 public:
  LinearForm(std::function<double(double, double)> f, bool apply_quadrature,
             size_t quad_order, bool dirichlet_boundary = true)
      : f_(f),
        apply_quadrature_(apply_quadrature),
        quad_order_(quad_order),
        dirichlet_boundary_(dirichlet_boundary) {}

  // Two methods of applying this linear form, using quadrature or
  // interpolation.
  template <typename I>
  void ApplyQuadrature(I *root);

  template <typename I>
  void ApplyInterpolation(I *root);

  // Resort to default setting.
  template <typename I>
  void Apply(I *root) {
    if (apply_quadrature_)
      ApplyQuadrature(root);
    else
      ApplyInterpolation(root);
  }

  size_t QuadratureOrder() const { return quad_order_; }
  const auto &Function() const { return f_; }

 protected:
  std::function<double(double, double)> f_;
  size_t quad_order_;
  bool dirichlet_boundary_;
  bool apply_quadrature_;

  // Evaluate the inner product between hat functions and f on elem.
  std::array<double, 3> QuadEval(Element2D *elem) const;
};

}  // namespace space

#include "linear_form.ipp"
