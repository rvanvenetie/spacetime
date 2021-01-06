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
  constexpr static const char *name = "Hierarch";

  explicit HierarchicalWaveletFn(
      const std::vector<HierarchicalWaveletFn *> parents, int index,
      SparseVector<ContLinearScalingFn> &&single_scale)
      : WaveletFn(parents, index, std::move(single_scale)) {
    assert(single_scale_.size() == 1);
    if (this->level_ > 0)
      for (auto &elem : support()) {
        assert(elem->psi_hierarch_ == nullptr);
        elem->psi_hierarch_ = this;
      }
  }

  bool is_full() const;
  bool Refine();
  double vertex() const {
    if (this->level_ == 0)
      return this->index_;
    else
      return (2 * this->index_ + 1) * std::pow(2, -this->level_);
  }

 protected:
  // Protected constructor for creating a metaroot.
  HierarchicalWaveletFn(
      datastructures::TreeContainer<HierarchicalWaveletFn> *container,
      const SmallVector<
          ContLinearScalingFn *,
          datastructures::NodeTrait<ContLinearScalingFn>::N_children>
          &mother_scalings);

  friend datastructures::Tree<HierarchicalWaveletFn>;
};
}  // namespace Time
