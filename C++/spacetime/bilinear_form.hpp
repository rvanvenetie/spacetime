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
  BilinearForm(DoubleTreeVector<BasisTimeIn, BasisSpace> *vec_in,
               DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out,
               bool use_cache = true);
  void Apply();
  void ApplyTranspose();

  const DoubleTreeVector<BasisTimeIn, BasisSpace> &sigma() { return sigma_; }
  const DoubleTreeVector<BasisTimeOut, BasisSpace> &theta() { return theta_; }

 protected:
  DoubleTreeVector<BasisTimeIn, BasisSpace> *vec_in_;
  DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out_;

  DoubleTreeVector<BasisTimeIn, BasisSpace> sigma_;
  DoubleTreeVector<BasisTimeOut, BasisSpace> theta_;
  bool use_cache_;
  bool is_cached_ = false;
  bool is_tcached_ = false;

  // Define frozen templates, useful for storing the bil forms.
  template <size_t i>
  using FI = datastructures::FrozenDoubleNode<
      datastructures::DoubleNodeVector<BasisTimeIn, BasisSpace>, i>;
  template <size_t i>
  using FO = datastructures::FrozenDoubleNode<
      datastructures::DoubleNodeVector<BasisTimeOut, BasisSpace>, i>;

  // Calculate th

  // Store bilinear forms in vectors.
  std::vector<space::BilinearForm<OperatorSpace, FI<1>, FI<1>>> bil_space_low_;
  std::vector<Time::BilinearForm<OperatorTime, FI<0>, FO<0>>> bil_time_low_;
  std::vector<Time::BilinearForm<OperatorTime, FI<0>, FO<0>>> bil_time_upp_;
  std::vector<space::BilinearForm<OperatorSpace, FO<1>, FO<1>>> bil_space_upp_;

  // Store bilinear forms of the transpose operations.
  std::vector<Time::BilinearForm<OperatorTime, FO<0>, FI<0>>> tbil_time_low_;
  std::vector<space::BilinearForm<OperatorSpace, FI<1>, FI<1>>> tbil_space_low_;
  std::vector<space::BilinearForm<OperatorSpace, FO<1>, FO<1>>> tbil_space_upp_;
  std::vector<Time::BilinearForm<OperatorTime, FO<0>, FI<0>>> tbil_time_upp_;
};

// Helper function.
template <template <typename, typename> class OpTime, typename OpSpace,
          typename BTimeIn, typename BTimeOut>
BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut> CreateBilinearForm(
    datastructures::DoubleTreeVector<BTimeIn, space::HierarchicalBasisFn>
        *vec_in,
    datastructures::DoubleTreeVector<BTimeOut, space::HierarchicalBasisFn>
        *vec_out,
    bool use_cache = true) {
  return BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>(
      vec_in, vec_out, use_cache = use_cache);
}

}  // namespace spacetime

#include "bilinear_form.ipp"
