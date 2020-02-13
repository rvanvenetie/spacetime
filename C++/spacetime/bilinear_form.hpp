#pragma once
#include <utility>
#include <vector>

#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"
#include "../space/bilinear_form.hpp"
#include "../time/bilinear_form.hpp"
#include "basis.hpp"

namespace spacetime {

// Forward declare.
template <typename BilForm>
class TransposeBilinearForm;

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
class BilinearForm {
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
};  // namespace spacetime

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
    return v;
  }

  auto vec_in() const { return a_->vec_in(); }
  auto vec_out() const { return a_->vec_out(); }

 protected:
  std::shared_ptr<BilFormA> a_;
  std::shared_ptr<BilFormB> b_;
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

// Creates the SumBilinearForm of two shared_ptrs.
template <typename BilFormA, typename BilFormB>
auto Sum(std::shared_ptr<BilFormA> a, std::shared_ptr<BilFormB> b) {
  return std::make_shared<SumBilinearForm<BilFormA, BilFormB>>(a, b);
}

}  // namespace spacetime

#include "bilinear_form.ipp"
