
#pragma once
#include <memory>
#include <vector>

#include "basis.hpp"
namespace Time {
class ContLinearScalingFn;
class ThreePointWaveletFn;

template <>
struct FunctionTrait<ContLinearScalingFn> {
  using Wavelet = ThreePointWaveletFn;
};

template <>
struct FunctionTrait<ThreePointWaveletFn> {
  using Scaling = ContLinearScalingFn;
};

class ContLinearScalingFn : public ScalingFn<ContLinearScalingFn> {
 public:
  constexpr static size_t order = 1;
  constexpr static bool continuous = true;
  constexpr static size_t N_children = 3;
  constexpr static size_t N_parents = 2;

  explicit ContLinearScalingFn(const std::vector<ContLinearScalingFn *> parents,
                               int index,
                               const std::vector<Element1D *> support)
      : ScalingFn<ContLinearScalingFn>(parents, index, support) {
    if (index > 0) {
      assert(!support[0]->phi_cont_lin_[1]);
      support[0]->phi_cont_lin_[1] = this;
    }
    if (index < (1 << level())) {
      assert(!support.back()->phi_cont_lin_[0]);
      support.back()->phi_cont_lin_[0] = this;
    }
  }

  double EvalMother(double t, bool deriv) const final;

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
  ContLinearScalingFn();

  friend datastructures::Tree<ContLinearScalingFn>;
  friend ThreePointWaveletFn;
  friend Element1D;
};

class ThreePointWaveletFn : public WaveletFn<ThreePointWaveletFn> {
 public:
  constexpr static size_t N_children = 2;
  constexpr static size_t N_parents = 2;

  explicit ThreePointWaveletFn(
      const std::vector<ThreePointWaveletFn *> parents, int index,
      const SparseVector<ContLinearScalingFn> &single_scale)
      : WaveletFn(parents, index, single_scale) {}

  bool is_full() const;
  bool Refine();

 protected:
  // Protected constructor for creating a metaroot.
  ThreePointWaveletFn();

  friend datastructures::Tree<ThreePointWaveletFn>;
};

// Define static variables.
extern datastructures::Tree<ContLinearScalingFn> cont_lin_tree;
extern datastructures::Tree<ThreePointWaveletFn> three_point_tree;

}  // namespace Time
