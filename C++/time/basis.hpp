#pragma once
#include <cmath>
#include <iostream>
#include <memory>

#include "../datastructures/tree.hpp"
namespace Time {

class Element1D : public datastructures::BinaryNode<Element1D> {
 public:
  // Constructors given the parent.
  explicit Element1D(Element1D *parent, int index)
      : BinaryNode(parent), index_(index) {}
  explicit Element1D(Element1D *parent, bool left_child)
      : Element1D(parent, parent->index() * 2 + (left_child ? 0 : 1)) {}

  int index() const { return index_; }

  bool Refine();
  std::pair<double, double> Interval() const;

 protected:
  // Protected constructor for creating a metaroot.
  Element1D() : BinaryNode(), index_(0) {
    children_.push_back(std::make_shared<Element1D>(this, 0));
  }

  int index_;

  friend datastructures::Tree<Element1D>;
};

template <typename I>
class Function : public datastructures::Node<I> {
 public:
  explicit Function(const std::vector<I *> &parents, int index,
                    const std::vector<Element1D *> &support = {})
      : datastructures::Node<I>(parents), index_(index), support_(support) {
    for (size_t i = 0; i < support_.size(); ++i) {
      assert(support[i]->level() == I::level_);
      if (i > 0) assert(support_[i - 1]->index() + 1 == support_[i]->index());
    }
  }

  inline std::pair<int, int> labda() const { return {I::level_, I::index_}; }
  inline int index() const { return index_; }
  const std::vector<Element1D *> &support() const { return support_; }

  virtual double Eval(double t, bool deriv = false) = 0;

 protected:
  // Protected constructor for creating a metaroot.
  Function() : datastructures::Node<I>(), index_(0) {}

  // The index inside this level.
  int index_;
  std::vector<Element1D *> support_;
};

template <typename I>
struct FunctionTrait;

template <typename I>
class ScalingFn : public Function<I> {
 public:
  // This is the transpose of wavelet -> single scale.
  std::vector<std::pair<typename FunctionTrait<I>::Wavelet *, double>>
      multi_scale_;

  double Eval(double t, bool deriv = false) override {
    int l = I::level_;
    int n = I::index_;
    double chain_rule_constant = deriv ? std::pow(2, l) : 1;
    return chain_rule_constant * EvalMother(std::pow(2, l) * t - n, deriv);
  }

  // To be implemented by derived classes.
  virtual double EvalMother(double t, bool deriv) = 0;

 protected:
  using Function<I>::Function;
};

template <typename I>
class WaveletFn : public Function<I> {
 public:
  using ScalingType = typename FunctionTrait<I>::Scaling;

  explicit WaveletFn(
      const std::vector<I *> &parents, int index,
      const std::vector<std::pair<ScalingType *, double>> &single_scale)
      : Function<I>(parents, index), single_scale_(single_scale) {
    // Now do two things:
    // 1) Register this wavelet in the scaling fn's multi_scale_.
    // 2) Figure out the support of this Wavelet using the single scale repr.

    for (int i = 0; i < single_scale_.size(); ++i) {
      // Sanity check.
      if (i > 0) {
        assert(single_scale_[i - 1].first->index() + 1 ==
               single_scale_[i].first->index());
      }

      auto [phi, coeff] = single_scale_[i];
      phi->multi_scale_.emplace_back(static_cast<I *>(this), coeff);
      for (auto elem : phi->support()) {
        support_.push_back(elem);
        assert(elem->level() == I::level_);
      }
    }

    // Deduplicate the support.
    std::sort(support_.begin(), support_.end(), [](auto elem1, auto elem2) {
      return elem1->index() < elem2->index();
    });
    auto last = std::unique(support_.begin(), support_.end());
    support_.erase(last, support_.end());
  }

  double Eval(double t, bool deriv = false) final {
    double result = 0;
    for (auto [fn, coeff] : single_scale_) {
      result += coeff * fn->Eval(t, deriv);
    }
    return result;
  }

  // This maps a wavelet to its single scale representation.
  std::vector<std::pair<ScalingType *, double>> single_scale_;

 protected:
  using Function<I>::Function;

  using Function<I>::support_;
};

// Declare static variables.
extern Element1D *mother_element;
extern datastructures::Tree<Element1D> elem_tree;

}  // namespace Time
