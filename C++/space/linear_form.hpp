#pragma once
#include "datastructures/include.hpp"

#include <stddef.h>
#include <utility>

namespace space {

class LinearFunctional {
 public:
  virtual ~LinearFunctional() {}
  virtual std::array<double, 3> Eval(Element2D *elem) const = 0;
};

template <size_t order>
class QuadratureFunctional : public LinearFunctional {
 public:
  QuadratureFunctional(std::function<double(double, double)> f) : f_(f) {}
  std::array<double, 3> Eval(Element2D *elem) const final;

 protected:
  std::function<double(double, double)> f_;
};

class LinearForm {
 public:
  LinearForm(std::unique_ptr<LinearFunctional> &&functional,
             bool dirichlet_boundary = true)
      : functional_(std::move(functional)),
        dirichlet_boundary_(dirichlet_boundary) {}

  template <typename I>
  void Apply(I *root);

 protected:
  std::unique_ptr<LinearFunctional> functional_;
  bool dirichlet_boundary_;
};

}  // namespace space

#include "linear_form.ipp"
