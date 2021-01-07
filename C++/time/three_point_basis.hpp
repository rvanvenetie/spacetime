
#pragma once
#include <memory>
#include <vector>

#include "basis.hpp"
namespace Time {
template <>
struct FunctionTrait<ThreePointWaveletFn> {
  using Scaling = ContLinearScalingFn;
};

class ContLinearScalingFn : public ScalingFn<ContLinearScalingFn> {
 public:
  constexpr static size_t order = 1;
  constexpr static bool continuous = true;
  constexpr static const char *name = "CLS";

  explicit ContLinearScalingFn(const std::vector<ContLinearScalingFn *> parents,
                               int index,
                               const std::vector<Element1D *> support)
      : ScalingFn<ContLinearScalingFn>(parents, index, support) {
    if (index > 0) {
      assert(!support[0]->phi_cont_lin_[1]);
      support[0]->phi_cont_lin_[1] = this;
    }
    if (index < (1LL << level())) {
      assert(!support.back()->phi_cont_lin_[0]);
      support.back()->phi_cont_lin_[0] = this;
    }
  }

  double EvalMother(double t, bool deriv) const;

  ContLinearScalingFn *RefineMiddle();
  ContLinearScalingFn *RefineLeft();
  ContLinearScalingFn *RefineRight() {
    if (child_right_) return child_right_;
    assert(nbr_right_);
    return nbr_right_->RefineLeft();
  }

 protected:
  ContLinearScalingFn *nbr_left_ = nullptr;
  ContLinearScalingFn *nbr_right_ = nullptr;

  ContLinearScalingFn *child_left_ = nullptr;
  ContLinearScalingFn *child_middle_ = nullptr;
  ContLinearScalingFn *child_right_ = nullptr;

  // Protected constructor for creating a metaroot.
  ContLinearScalingFn(Deque<ContLinearScalingFn> *container,
                      Element1D *mother_element);

  friend datastructures::Tree<ContLinearScalingFn>;
  friend ThreePointWaveletFn;
  friend HierarchicalWaveletFn;
  friend Element1D;
};

class ThreePointWaveletFn : public WaveletFn<ThreePointWaveletFn> {
 public:
  constexpr static const char *name = "Three";

  explicit ThreePointWaveletFn(const std::vector<ThreePointWaveletFn *> parents,
                               int index,
                               SparseVector<ContLinearScalingFn> &&single_scale)
      : WaveletFn(parents, index, std::move(single_scale)) {}

  bool is_full() const;
  bool Refine();

 protected:
  // Protected constructor for creating a metaroot.
  ThreePointWaveletFn(
      Deque<ThreePointWaveletFn> *container,
      const SmallVector<
          ContLinearScalingFn *,
          datastructures::NodeTrait<ContLinearScalingFn>::N_children>
          &mother_scalings);

  friend datastructures::Tree<ThreePointWaveletFn>;
};
}  // namespace Time
