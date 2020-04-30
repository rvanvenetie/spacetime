#pragma once
#include <memory>
#include <vector>

#include "basis.hpp"

namespace Time {

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
  constexpr static const char *name = "DCS";

  explicit DiscConstantScalingFn(DiscConstantScalingFn *parent, int index,
                                 Element1D *support)
      : ScalingFn<DiscConstantScalingFn>({parent}, index, {support}) {
    assert(!support->phi_disc_const_);
    support->phi_disc_const_ = this;
  }

  double EvalMother(double t, bool deriv) const;
  bool Refine();

 protected:
  // Protected constructor for creating a metaroot.
  DiscConstantScalingFn(std::deque<DiscConstantScalingFn> *container,
                        Element1D *mother_element);
  inline bool is_full() const {
    if (is_metaroot())
      return children_.size() == 1;
    else
      return children_.size() == 2;
  }

  friend datastructures::Tree<DiscConstantScalingFn>;
  friend HaarWaveletFn;
  friend Element1D;
};

class HaarWaveletFn : public WaveletFn<HaarWaveletFn> {
 public:
  constexpr static const char *name = "Haar";

  explicit HaarWaveletFn(HaarWaveletFn *parent, int index,
                         SparseVector<DiscConstantScalingFn> &&single_scale)
      : WaveletFn({parent}, index, std::move(single_scale)) {}

  bool Refine();
  bool is_full() const;

 protected:
  // Protected constructor for creating a metaroot.
  HaarWaveletFn(std::deque<HaarWaveletFn> *container,
                DiscConstantScalingFn *mother_scaling);

  friend datastructures::Tree<HaarWaveletFn>;
};
}  // namespace Time
