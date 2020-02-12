#pragma once
#include <utility>
#include <vector>

#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"
#include "../space/bilinear_form.hpp"
#include "../time/bilinear_form.hpp"

namespace spacetime {

// Forward declare.
template <typename BilForm>
class TransposeBilinearForm;

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
  BilinearForm(DoubleTreeVector<BasisTimeIn, BasisSpace> *vec_in,
               DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out,
               bool use_cache = true);

  Eigen::VectorXd Apply();
  Eigen::VectorXd ApplyTranspose();

  // Returns the transpose operator.
  auto Transpose() const { return TransposeBilinearForm(shared_from_this()); }

  auto vec_in() const { return vec_in_; }
  auto vec_out() const { return vec_out_; }
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
std::shared_ptr<BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>>
CreateBilinearForm(
    datastructures::DoubleTreeVector<BTimeIn, space::HierarchicalBasisFn>
        *vec_in,
    datastructures::DoubleTreeVector<BTimeOut, space::HierarchicalBasisFn>
        *vec_out,
    bool use_cache = true) {
  return std::make_shared<BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>>(
      vec_in, vec_out, use_cache = use_cache);
}

// This class represents the sum of two bilinear forms.
template <typename BilFormA, typename BilFormB = BilFormA>
class SumBilinearForm {
 public:
  SumBilinearForm(std::shared_ptr<BilFormA> a, std::shared_ptr<BilFormB> b)
      : a_(a), b_(b) {
    assert(a->vec_in() == b->vec_in());
    assert(a->vec_out() == b->vec_out());
  }

  Eigen::VectorXd Apply() {
    // Apply both operators and store the vectorized result.
    auto v = a_->Apply();
    v += b_->Apply();

    a_->vec_out()->FromVectorContainer(v);
    return v;
  }

  auto vec_in() const { return a_->vec_in(); }
  auto vec_out() const { return a_->vec_out(); }

 protected:
  std::shared_ptr<BilFormA> a_;
  std::shared_ptr<BilFormB> b_;
};

// This class represents the transpose of a spacetime bilinear form.
template <typename BilForm>
class TransposeBilinearForm {
 public:
  TransposeBilinearForm(std::shared_ptr<BilForm> bil_form)
      : bil_form_(bil_form) {
    assert(bil_form_->use_cache_);
  }

  Eigen::VectorXd Apply() { return bil_form_->ApplyTranspose(); }

  auto vec_in() const { return bil_form_->vec_in(); }
  auto vec_out() const { return bil_form_->vec_out(); }

 protected:
  std::shared_ptr<BilForm> bil_form_;
};

}  // namespace spacetime

#include "bilinear_form.ipp"
