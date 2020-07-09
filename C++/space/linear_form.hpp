#pragma once
#include <stddef.h>

#include <utility>

#include "triangulation.hpp"

namespace space {

class QuadratureFunctional {
 public:
  QuadratureFunctional(std::function<double(double, double)> f, size_t order)
      : f_(f), order_(order) {}
  std::array<double, 3> Eval(Element2D *elem) const;

  std::function<double(double, double)> Function() const { return f_; }
  size_t Order() const { return order_; }

 protected:
  std::function<double(double, double)> f_;
  size_t order_;
};

class LinearForm {
 public:
  LinearForm(std::unique_ptr<QuadratureFunctional> &&functional,
             bool dirichlet_boundary = true)
      : functional_(std::move(functional)),
        dirichlet_boundary_(dirichlet_boundary) {}

  template <typename I>
  void Apply(I *root);

  QuadratureFunctional *Functional() const { return functional_.get(); }

 protected:
  std::unique_ptr<QuadratureFunctional> functional_;
  bool dirichlet_boundary_;
};
}  // namespace space

#include "linear_form.ipp"
