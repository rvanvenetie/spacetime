#pragma once
#include <utility>
#include <vector>

#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"
#include "../space/bilinear_form.hpp"
#include "../time/bilinear_form.hpp"

namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
class BilinearForm {
 protected:
  template <typename T0, typename T1>
  using DoubleTreeVector = datastructures::DoubleTreeVector<T0, T1>;
  using BasisSpace = space::HierarchicalBasisFn;

 public:
  BilinearForm(const DoubleTreeVector<BasisTimeIn, BasisSpace> &vec_in,
               DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out,
               bool use_cache = true);
  void Apply();

 protected:
  const DoubleTreeVector<BasisTimeIn, BasisSpace> &vec_in_;
  DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out_;
  DoubleTreeVector<BasisTimeOut, BasisSpace> vec_out_low_;

  DoubleTreeVector<BasisTimeIn, BasisSpace> sigma_;
  DoubleTreeVector<BasisTimeOut, BasisSpace> theta_;
  bool use_cache_;

  // The code below is necessary for caching the bilinear_forms.
  template <size_t i>
  using FrozenIn = datastructures::FrozenDoubleNode<
      datastructures::MultiNodeVector<BasisTimeIn, BasisSpace>, i>;
  template <size_t i>
  using FrozenOut = datastructures::FrozenDoubleNode<
      datastructures::MultiNodeVector<BasisTimeOut, BasisSpace>, i>;

  // Store bilinear forms in vectors.
  std::vector<space::BilinearForm<OperatorSpace, FrozenIn<1>, FrozenIn<1>>>
      bil_space_low_;
  std::vector<Time::BilinearForm<OperatorTime, FrozenIn<0>, FrozenOut<0>>>
      bil_time_low_;
  std::vector<Time::BilinearForm<OperatorTime, FrozenIn<0>, FrozenOut<0>>>
      bil_time_upp_;
  std::vector<space::BilinearForm<OperatorSpace, FrozenOut<1>, FrozenOut<1>>>
      bil_space_upp_;
};

// Helper function.
template <template <typename, typename> class OpTime, typename OpSpace,
          typename BTimeIn, typename BTimeOut>
BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut> CreateBilinearForm(
    const datastructures::DoubleTreeVector<BTimeIn, space::HierarchicalBasisFn>
        &vec_in,
    datastructures::DoubleTreeVector<BTimeOut, space::HierarchicalBasisFn>
        *vec_out,
    bool use_cache = true) {
  return BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>(
      vec_in, vec_out, use_cache = use_cache);
}

}  // namespace spacetime

#include "bilinear_form.ipp"
