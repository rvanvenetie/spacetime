#pragma once

#include <functional>

#include "integration.hpp"
#include "sparse_vector.hpp"
namespace Time {

template <typename Basis>
class LinearFunctional {
 public:
  virtual ~LinearFunctional<Basis>() {}

  SparseVector<Basis> Eval(const SparseIndices<Basis> &indices) const {
    SparseVector<Basis> out;
    out.reserve(indices.size());
    for (auto phi : indices) out.emplace_back(phi, Eval(phi));
    return out;
  }
  virtual double Eval(Basis *phi) const = 0;
};

template <typename Basis>
class QuadratureFunctional : public LinearFunctional<Basis> {
 public:
  using LinearFunctional<Basis>::Eval;

  QuadratureFunctional(std::function<double(double)> f, size_t order)
      : f_(f), order_(order) {}

  double Eval(Basis *phi) const {
    double cell = 0.0;
    for (auto elem : phi->support())
      cell += Integrate([&](double t) { return f_(t) * phi->Eval(t); }, *elem,
                        order_ + 1);
    return cell;
  }

 protected:
  std::function<double(double)> f_;
  size_t order_;
};

template <typename Basis>
class ZeroEvalFunctional : public LinearFunctional<Basis> {
 public:
  using LinearFunctional<Basis>::Eval;

  double Eval(Basis *phi) const { return phi->Eval(0.0); }
};
};  // namespace Time
