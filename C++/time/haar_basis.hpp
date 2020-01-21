#pragma once
#include <memory>
#include <vector>

#include "basis.hpp"

namespace Time {
class DiscConstantScalingFn;
class HaarWaveletFn;

template <>
struct FunctionTrait<DiscConstantScalingFn> {
  using Wavelet = HaarWaveletFn;
};

template <>
struct FunctionTrait<HaarWaveletFn> {
  using Scaling = DiscConstantScalingFn;
};

class DiscConstantScalingFn : public ScalingFn<DiscConstantScalingFn> {
 public:
  constexpr static size_t order = 0;
  constexpr static bool continuous = false;
  constexpr static size_t N_children = 2;

  explicit DiscConstantScalingFn(DiscConstantScalingFn *parent, int index,
                                 Element1D *support)
      : ScalingFn<DiscConstantScalingFn>({parent}, index, {support}) {}

  double EvalMother(double t, bool deriv) final;
  bool Refine();

 protected:
  // Protected constructor for creating a metaroot.
  DiscConstantScalingFn();
  inline bool is_full() const {
    if (is_metaroot())
      return children_.size() == 1;
    else
      return children_.size() == 2;
  }

  friend datastructures::Tree<DiscConstantScalingFn>;
  friend HaarWaveletFn;
};

class HaarWaveletFn : public WaveletFn<HaarWaveletFn> {
 public:
  constexpr static size_t N_children = 2;

  explicit HaarWaveletFn(
      HaarWaveletFn *parent, int index,
      const std::vector<std::pair<DiscConstantScalingFn *, double>>
          &single_scale)
      : WaveletFn({parent}, index, single_scale) {}

  bool Refine();
  bool is_full() const;

 protected:
  // Protected constructor for creating a metaroot.
  HaarWaveletFn();

  friend datastructures::Tree<HaarWaveletFn>;
};

// Define static variables.
extern datastructures::Tree<DiscConstantScalingFn> disc_cons_tree;
extern datastructures::Tree<HaarWaveletFn> haar_tree;

}  // namespace Time
