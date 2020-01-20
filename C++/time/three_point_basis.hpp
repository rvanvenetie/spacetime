
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
  bool is_full() const;

  explicit ContLinearScalingFn(const std::vector<ContLinearScalingFn *> parents,
                               int index,
                               const std::vector<Element1D *> support)
      : ScalingFn<ContLinearScalingFn>(parents, index, support) {}

  double EvalMother(double t, bool deriv) final;
  bool Refine();

 protected:
  ContLinearScalingFn *nbr_left_ = nullptr;
  ContLinearScalingFn *nbr_right_ = nullptr;

  ContLinearScalingFn *child_left_ = nullptr;
  ContLinearScalingFn *child_mid_ = nullptr;
  ContLinearScalingFn *child_right_ = nullptr;

  // Protected constructor for creating a metaroot.
  ContLinearScalingFn();

  friend datastructures::Tree<ContLinearScalingFn>;
  friend ThreePointWaveletFn;
};

class ThreePointWaveletFn : public WaveletFn<ThreePointWaveletFn> {
 public:
  constexpr static size_t N_children = 2;
  bool is_full() const;

  explicit ThreePointWaveletFn(
      ThreePointWaveletFn *parent, int index,
      const std::vector<std::pair<ContLinearScalingFn *, double>> &single_scale)
      : WaveletFn({parent}, index, single_scale) {}

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
