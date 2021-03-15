#pragma once

#include "linear_functional.hpp"

namespace Time {
template <typename WaveletBasis>
class LinearForm {
 public:
  using ScalingBasis = typename FunctionTrait<WaveletBasis>::Scaling;

  LinearForm(std::unique_ptr<LinearFunctional<ScalingBasis>> functional)
      : functional_(std::move(functional)) {}

  template <typename I>
  void Apply(I *root);

 protected:
  std::unique_ptr<LinearFunctional<ScalingBasis>> functional_;
  std::vector<SparseIndices<WaveletBasis>> lvl_ind_out_;
  SparseIndices<WaveletBasis> empty_ind_out_;

  std::pair<SparseVector<ScalingBasis>, SparseVector<WaveletBasis>> ApplyRecur(
      size_t l, const SparseIndices<ScalingBasis> &Pi_out);

  std::pair<SparseIndices<ScalingBasis>, SparseIndices<ScalingBasis>>
  ConstructPiOut(const SparseIndices<ScalingBasis> &Pi_out);
};
}  // namespace Time

#include "linear_form.ipp"
