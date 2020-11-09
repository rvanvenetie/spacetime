#pragma once
#include <memory>
#include <vector>

#include "basis.hpp"
#include "three_point_basis.hpp"
namespace Time {
template <>
struct FunctionTrait<HierarchicalWaveletFn> {
  using Scaling = ContLinearScalingFn;
};

class HierarchicalWaveletFn : public WaveletFn<HierarchicalWaveletFn> {
 public:
  constexpr static const char *name = "Three";

  explicit HierarchicalWaveletFn(
      const std::vector<HierarchicalWaveletFn *> parents, int index,
      SparseVector<ContLinearScalingFn> &&single_scale)
      : WaveletFn(parents, index, std::move(single_scale)) {
    assert(single_scale_.size() == 1);
  }

  bool is_full() const;
  bool Refine();

 protected:
  // Protected constructor for creating a metaroot.
  HierarchicalWaveletFn(
      Deque<HierarchicalWaveletFn> *container,
      const SmallVector<
          ContLinearScalingFn *,
          datastructures::NodeTrait<ContLinearScalingFn>::N_children>
          &mother_scalings);

  friend datastructures::Tree<HierarchicalWaveletFn>;
};
}  // namespace Time
