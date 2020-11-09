#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <memory>

#include "../datastructures/tree.hpp"
#include "sparse_vector.hpp"

// Forward all the class declarations.
namespace Time {
class Element1D;
class DiscConstantScalingFn;
class HaarWaveletFn;
class DiscLinearScalingFn;
class OrthonormalWaveletFn;
class ContLinearScalingFn;
class ThreePointWaveletFn;
class HierarchicalWaveletFn;
}  // namespace Time

namespace datastructures {
template <>
struct NodeTrait<Time::Element1D> {
  static constexpr size_t N_parents = 1;
  static constexpr size_t N_children = 2;
};
template <>
struct NodeTrait<Time::DiscConstantScalingFn> {
  static constexpr size_t N_parents = 1;
  static constexpr size_t N_children = 2;
};
template <>
struct NodeTrait<Time::HaarWaveletFn> {
  static constexpr size_t N_parents = 1;
  static constexpr size_t N_children = 2;
};
template <>
struct NodeTrait<Time::DiscLinearScalingFn> {
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 4;
};
template <>
struct NodeTrait<Time::OrthonormalWaveletFn> {
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 4;
};
template <>
struct NodeTrait<Time::ContLinearScalingFn> {
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 3;
};
template <>
struct NodeTrait<Time::ThreePointWaveletFn> {
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 2;
};
template <>
struct NodeTrait<Time::HierarchicalWaveletFn> {
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 2;
};
}  // namespace datastructures

namespace Time {
class Element1D : public datastructures::BinaryNode<Element1D> {
 public:
  // Constructors given the parent.
  explicit Element1D(Element1D *parent, int index)
      : BinaryNode(parent), index_(index) {}
  explicit Element1D(Element1D *parent, bool left_child)
      : Element1D(parent, parent->index() * 2 + (left_child ? 0 : 1)) {}

  int index() const { return index_; }

  // Refines the actual element.
  bool Refine();

  // Ensures that the given basis functions exist on the element.
  const std::array<ContLinearScalingFn *, 2> &RefineContLinear();

  const std::array<DiscLinearScalingFn *, 2> &PhiDiscLinear() {
    return phi_disc_lin_;
  }

  const std::array<OrthonormalWaveletFn *, 2> &RefinePsiOrthonormal();

  std::pair<double, double> Interval() const;
  double GlobalCoordinates(double bary2) const;
  double area() const {
    auto [a, b] = Interval();
    return b - a;
  }

  friend std::ostream &operator<<(std::ostream &os, const Element1D &elem) {
    os << "I(" << elem.Interval().first << ", " << elem.Interval().second
       << ")";
    return os;
  }

 protected:
  // Protected constructor for creating a metaroot.
  Element1D(Deque<Element1D> *container) : BinaryNode(container), index_(0) {
    make_child(/* parent */ this, /* index */ 0);
  }

  int index_;

  // There is a mapping between the element and the basis functions.
  DiscConstantScalingFn *phi_disc_const_ = nullptr;
  std::array<ContLinearScalingFn *, 2> phi_cont_lin_ = {nullptr, nullptr};
  std::array<DiscLinearScalingFn *, 2> phi_disc_lin_ = {nullptr, nullptr};
  std::array<OrthonormalWaveletFn *, 2> psi_ortho_ = {nullptr, nullptr};

  friend DiscConstantScalingFn;
  friend ContLinearScalingFn;
  friend DiscLinearScalingFn;
  friend OrthonormalWaveletFn;
  friend datastructures::Tree<Element1D>;
};

template <typename I>
class Function : public datastructures::Node<I> {
 public:
  explicit Function(const std::vector<I *> &parents, int index,
                    const std::vector<Element1D *> &support = {})
      : datastructures::Node<I>(parents), index_(index), support_(support) {
    for (size_t i = 0; i < support_.size(); ++i) {
      assert(support[i]->level() == this->level_);
      if (i > 0) assert(support_[i - 1]->index() + 1 == support_[i]->index());
    }
  }

  inline std::pair<int, int> labda() const {
    return {this->level_, this->index_};
  }
  inline int index() const { return index_; }
  const std::vector<Element1D *> &support() const { return support_; }
  double center() const {
    return (support_[0]->Interval().first +
            support_.back()->Interval().second) /
           2.0;
  }
  std::pair<double, double> Interval() const {
    return {support_[0]->Interval().first, support_.back()->Interval().second};
  }

  friend std::ostream &operator<<(std::ostream &os, const Function<I> &fn) {
    os << I::name << "(" << fn.level() << ", " << fn.index() << ")";
    return os;
  }

 protected:
  // Protected constructor for creating a metaroot.
  Function(Deque<I> *container)
      : datastructures::Node<I>(container), index_(0) {}

  // The index inside this level.
  int index_;

  // The vector of elements that make up this functions support.
  std::vector<Element1D *> support_;
};

template <typename I>
struct FunctionTrait;

template <typename I>
class WaveletFn;

template <typename I>
class ScalingFn : public Function<I> {
 public:
  double Eval(double t, bool deriv = false) const {
    int l = this->level_;
    int n = this->index_;
    double chain_rule_constant = deriv ? (1 << l) : 1;
    return chain_rule_constant *
           static_cast<const I &>(*this).EvalMother((1 << l) * t - n, deriv);
  }

 protected:
  using Function<I>::Function;
};

template <typename I>
class WaveletFn : public Function<I> {
 public:
  using ScalingType = typename FunctionTrait<I>::Scaling;

  explicit WaveletFn(const std::vector<I *> &parents, int index,
                     SparseVector<ScalingType> &&single_scale)
      : Function<I>(parents, index), single_scale_(std::move(single_scale)) {
    // Now do two things:
    // 1) Register this wavelet in the scaling fn's multi_scale_.
    // 2) Figure out the support of this Wavelet using the single scale repr.

    for (int i = 0; i < single_scale_.size(); ++i) {
      assert(single_scale_[i].first != nullptr);
      // Sanity check.
      if (i > 0) {
        assert(single_scale_[i - 1].first->index() <
               single_scale_[i].first->index());
      }

      auto [phi, _] = single_scale_[i];
      for (auto elem : phi->support()) {
        support_.push_back(elem);
        assert(elem->level() == this->level_);
      }
    }

    // Deduplicate the support.
    std::sort(support_.begin(), support_.end(), [](auto elem1, auto elem2) {
      return elem1->index() < elem2->index();
    });
    auto last = std::unique(support_.begin(), support_.end());
    support_.erase(last, support_.end());
  }

  double Eval(double t, bool deriv = false) const {
    double result = 0;
    for (auto [fn, coeff] : single_scale_) {
      result += coeff * fn->Eval(t, deriv);
    }
    return result;
  }

  const SparseVector<ScalingType> &single_scale() const {
    return single_scale_;
  }

 protected:
  using Function<I>::Function;
  using Function<I>::support_;

  // This maps a wavelet to its single scale representation.
  SparseVector<ScalingType> single_scale_;
};
}  // namespace Time
