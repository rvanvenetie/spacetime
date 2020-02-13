#pragma once
#include <utility>
#include <vector>

#include "../datastructures/bilinear_form.hpp"
#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"
#include "../space/bilinear_form.hpp"
#include "../time/bilinear_form.hpp"
#include "basis.hpp"

namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
class BilinearForm
    : public std::enable_shared_from_this<BilinearForm<
          OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>> {
 protected:
  template <typename T0, typename T1>
  using DoubleTreeVector = datastructures::DoubleTreeVector<T0, T1>;
  using BasisSpace = space::HierarchicalBasisFn;

 public:
  BilinearForm(
      DoubleTreeVector<BasisTimeIn, BasisSpace> *vec_in,
      DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out,
      std::shared_ptr<DoubleTreeVector<BasisTimeIn, BasisSpace>> sigma,
      std::shared_ptr<DoubleTreeVector<BasisTimeOut, BasisSpace>> theta,
      bool use_cache = true);

  Eigen::VectorXd Apply();
  Eigen::VectorXd ApplyTranspose();
  auto Transpose() {
    if (!transpose_)
      transpose_ =
          std::make_shared<typename decltype(transpose_)::element_type>(
              this->shared_from_this());
    return transpose_;
  }

  auto vec_in() const { return vec_in_; }
  auto vec_out() const { return vec_out_; }
  auto sigma() { return sigma_; }
  auto theta() { return theta_; }

 protected:
  DoubleTreeVector<BasisTimeIn, BasisSpace> *vec_in_;
  DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out_;

  std::shared_ptr<DoubleTreeVector<BasisTimeIn, BasisSpace>> sigma_;
  std::shared_ptr<DoubleTreeVector<BasisTimeOut, BasisSpace>> theta_;
  bool use_cache_;
  bool is_cached_ = false;
  std::shared_ptr<TransposeBilinearForm<
      BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>>>
      transpose_;

  // Define frozen templates, useful for storing the bil forms.
  template <size_t i>
  using FI = datastructures::FrozenDoubleNode<
      datastructures::DoubleNodeVector<BasisTimeIn, BasisSpace>, i>;
  template <size_t i>
  using FO = datastructures::FrozenDoubleNode<
      datastructures::DoubleNodeVector<BasisTimeOut, BasisSpace>, i>;

  // Store bilinear forms in vectors.
  std::vector<space::BilinearForm<OperatorSpace, FI<1>, FI<1>>> bil_space_low_;
  std::vector<Time::BilinearForm<OperatorTime, FI<0>, FO<0>>> bil_time_low_;
  std::vector<Time::BilinearForm<OperatorTime, FI<0>, FO<0>>> bil_time_upp_;
  std::vector<space::BilinearForm<OperatorSpace, FO<1>, FO<1>>> bil_space_upp_;
};

// Helper functions.
template <template <typename, typename> class OpTime, typename OpSpace,
          typename BTimeIn, typename BTimeOut>
std::shared_ptr<BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>>
CreateBilinearForm(
    datastructures::DoubleTreeVector<BTimeIn, space::HierarchicalBasisFn>
        *vec_in,
    datastructures::DoubleTreeVector<BTimeOut, space::HierarchicalBasisFn>
        *vec_out,
    bool use_cache = true) {
  return std::make_shared<BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>>(
      vec_in, vec_out, GenerateSigma(*vec_in, *vec_out),
      GenerateTheta(*vec_in, *vec_out), use_cache);
}

template <template <typename, typename> class OpTime, typename OpSpace,
          typename BTimeIn, typename BTimeOut>
std::shared_ptr<BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>>
CreateBilinearForm(
    datastructures::DoubleTreeVector<BTimeIn, space::HierarchicalBasisFn>
        *vec_in,
    datastructures::DoubleTreeVector<BTimeOut, space::HierarchicalBasisFn>
        *vec_out,
    std::shared_ptr<
        datastructures::DoubleTreeVector<BTimeIn, space::HierarchicalBasisFn>>
        sigma,
    std::shared_ptr<
        datastructures::DoubleTreeVector<BTimeOut, space::HierarchicalBasisFn>>
        theta,
    bool use_cache = true) {
  return std::make_shared<BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>>(
      vec_in, vec_out, sigma, theta, use_cache);
}

}  // namespace spacetime

#include "bilinear_form.ipp"
