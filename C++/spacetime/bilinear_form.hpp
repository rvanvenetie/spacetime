#pragma once
#include <utility>
#include <vector>

#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"
#include "../space/bilinear_form.hpp"
#include "../time/bilinear_form.hpp"
#include "basis.hpp"
#include "bilinear_form_linalg.hpp"

namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
class BilinearForm
    : public std::enable_shared_from_this<
          BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>>,
      public BilinearFormBase<datastructures::DoubleTreeVector<
                                  BasisTimeIn, space::HierarchicalBasisFn>,
                              datastructures::DoubleTreeVector<
                                  BasisTimeOut, space::HierarchicalBasisFn>> {
 protected:
  template <typename T0, typename T1>
  using DoubleTreeVector = datastructures::DoubleTreeVector<T0, T1>;
  using BasisSpace = space::HierarchicalBasisFn;

 public:
  // Friendly aliases.
  using DblVecIn = DoubleTreeVector<BasisTimeIn, BasisSpace>;
  using DblVecOut = DoubleTreeVector<BasisTimeOut, BasisSpace>;

  BilinearForm(DblVecIn *vec_in, DblVecOut *vec_out, bool use_cache,
               space::OperatorOptions space_opts = space::OperatorOptions());
  BilinearForm(
      DblVecIn *vec_in, DblVecOut *vec_out,
      std::shared_ptr<DoubleTreeVector<BasisTimeIn, BasisSpace>> sigma,
      std::shared_ptr<DoubleTreeVector<BasisTimeOut, BasisSpace>> theta,
      bool use_cache,
      space::OperatorOptions space_opts = space::OperatorOptions());

  // Apply takes data from vec_in and writes it to vec_out.
  Eigen::VectorXd Apply(const Eigen::VectorXd &v) final;
  DblVecIn *vec_in() const final { return vec_in_; }
  DblVecOut *vec_out() const final { return vec_out_; }

  // ApplyTranspose takes data from *vec_out* and writes it to *vec_in*.
  Eigen::VectorXd ApplyTranspose(const Eigen::VectorXd &v);

  auto Transpose() {
    return std::make_shared<TransposeBilinearForm<
        BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>>>(
        this->shared_from_this());
  }

  auto sigma() { return sigma_; }
  auto theta() { return theta_; }

 protected:
  // References to in/out vectors.
  DblVecIn *vec_in_;
  DblVecOut *vec_out_;

  // Options.
  bool use_cache_;
  space::OperatorOptions space_opts_;

  std::shared_ptr<DoubleTreeVector<BasisTimeIn, BasisSpace>> sigma_;
  std::shared_ptr<DoubleTreeVector<BasisTimeOut, BasisSpace>> theta_;

  // Debug information.
  using BilinearFormBase<DblVecIn, DblVecOut>::time_construct_;
  using BilinearFormBase<DblVecIn, DblVecOut>::time_apply_;
  using BilinearFormBase<DblVecIn, DblVecOut>::time_apply_split_;
  using BilinearFormBase<DblVecIn, DblVecOut>::num_apply_;

  // Define frozen templates, useful for storing the bil forms.
  template <size_t i>
  using FI = datastructures::FrozenDoubleNode<
      datastructures::DoubleNodeVector<BasisTimeIn, BasisSpace>, i>;
  template <size_t i>
  using FO = datastructures::FrozenDoubleNode<
      datastructures::DoubleNodeVector<BasisTimeOut, BasisSpace>, i>;

  // Store (cached) bilinear forms in vectors.
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
    bool use_cache) {
  return std::make_shared<BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>>(
      vec_in, vec_out, use_cache);
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
    bool use_cache) {
  return std::make_shared<BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>>(
      vec_in, vec_out, sigma, theta, use_cache);
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTime>
class SymmetricBilinearForm
    : public BilinearFormBase<datastructures::DoubleTreeVector<
                                  BasisTime, space::HierarchicalBasisFn>,
                              datastructures::DoubleTreeVector<
                                  BasisTime, space::HierarchicalBasisFn>> {
 protected:
  template <typename T0, typename T1>
  using DoubleTreeVector = datastructures::DoubleTreeVector<T0, T1>;
  using BasisSpace = space::HierarchicalBasisFn;

 public:
  // Friendly aliases.
  using DblVec = DoubleTreeVector<BasisTime, BasisSpace>;

  SymmetricBilinearForm(
      DblVec *vec, bool use_cache,
      space::OperatorOptions space_opts = space::OperatorOptions());

  // Apply takes data from vec and writes it to vec.
  Eigen::VectorXd Apply(const Eigen::VectorXd &v) final;
  DblVec *vec_in() const final { return vec_; }
  DblVec *vec_out() const final { return vec_; }

 protected:
  DblVec *vec_;

  // Options.
  bool use_cache_;
  space::OperatorOptions space_opts_;

  // Debug information.
  using BilinearFormBase<DblVec, DblVec>::time_construct_;
  using BilinearFormBase<DblVec, DblVec>::time_apply_;
  using BilinearFormBase<DblVec, DblVec>::time_apply_split_;
  using BilinearFormBase<DblVec, DblVec>::num_apply_;

  // Define frozen templates, useful for storing the bil forms.
  template <size_t i>
  using F = datastructures::FrozenDoubleNode<
      datastructures::DoubleNodeVector<BasisTime, BasisSpace>, i>;

  // Store (cached) bilinear forms in vectors.
  std::vector<space::BilinearForm<OperatorSpace, F<1>, F<1>>> bil_space_low_;
  std::vector<Time::BilinearForm<OperatorTime, F<0>, F<0>>> bil_time_low_;
};

template <typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
class BlockDiagonalBilinearForm
    : public BilinearFormBase<datastructures::DoubleTreeVector<
                                  BasisTimeIn, space::HierarchicalBasisFn>,
                              datastructures::DoubleTreeVector<
                                  BasisTimeOut, space::HierarchicalBasisFn>> {
 protected:
  template <typename T0, typename T1>
  using DoubleTreeVector = datastructures::DoubleTreeVector<T0, T1>;
  using BasisSpace = space::HierarchicalBasisFn;

 public:
  // Friendly aliases.
  using DblVecIn = DoubleTreeVector<BasisTimeIn, BasisSpace>;
  using DblVecOut = DoubleTreeVector<BasisTimeOut, BasisSpace>;

  BlockDiagonalBilinearForm(
      DblVecIn *vec_in, DblVecOut *vec_out, bool use_cache,
      space::OperatorOptions space_opts = space::OperatorOptions());

  // Apply takes data from vec_in and writes it to vec_out.
  Eigen::VectorXd Apply(const Eigen::VectorXd &v) final;
  DblVecIn *vec_in() const final { return vec_in_; }
  DblVecOut *vec_out() const final { return vec_out_; }

 protected:
  bool use_cache_;
  space::OperatorOptions space_opts_;

  DblVecIn *vec_in_;
  DblVecOut *vec_out_;

  // Debug information.
  using BilinearFormBase<DblVecIn, DblVecOut>::time_construct_;
  using BilinearFormBase<DblVecIn, DblVecOut>::time_apply_;
  using BilinearFormBase<DblVecIn, DblVecOut>::num_apply_;

  // The (cached) bilinear forms.
  template <size_t i>
  using FI = datastructures::FrozenDoubleNode<
      datastructures::DoubleNodeVector<BasisTimeIn, BasisSpace>, i>;
  std::vector<space::BilinearForm<OperatorSpace, FI<1>, FI<1>>> space_bilforms_;
};

template <typename OpSpace, typename BTimeIn, typename BTimeOut>
std::shared_ptr<BlockDiagonalBilinearForm<OpSpace, BTimeIn, BTimeOut>>
CreateBlockDiagonalBilinearForm(
    datastructures::DoubleTreeVector<BTimeIn, space::HierarchicalBasisFn>
        *vec_in,
    datastructures::DoubleTreeVector<BTimeOut, space::HierarchicalBasisFn>
        *vec_out,
    bool use_cache) {
  return std::make_shared<
      BlockDiagonalBilinearForm<OpSpace, BTimeIn, BTimeOut>>(vec_in, vec_out,
                                                             use_cache);
}

// Template specializations for faster compile times.
extern template class BilinearForm<Time::MassOperator, space::StiffnessOperator,
                                   Time::OrthonormalWaveletFn,
                                   Time::OrthonormalWaveletFn>;
extern template class BilinearForm<Time::TransportOperator, space::MassOperator,
                                   Time::ThreePointWaveletFn,
                                   Time::OrthonormalWaveletFn>;
extern template class BilinearForm<Time::MassOperator, space::StiffnessOperator,
                                   Time::ThreePointWaveletFn,
                                   Time::OrthonormalWaveletFn>;
extern template class BilinearForm<Time::ZeroEvalOperator, space::MassOperator,
                                   Time::ThreePointWaveletFn,
                                   Time::ThreePointWaveletFn>;
extern template class SymmetricBilinearForm<
    Time::MassOperator, space::StiffnessOperator, Time::OrthonormalWaveletFn>;
extern template class SymmetricBilinearForm<
    Time::ZeroEvalOperator, space::MassOperator, Time::ThreePointWaveletFn>;

extern template class BlockDiagonalBilinearForm<
    space::DirectInverse<space::StiffnessOperator>, Time::OrthonormalWaveletFn,
    Time::OrthonormalWaveletFn>;
extern template class BlockDiagonalBilinearForm<
    space::XPreconditionerOperator<space::DirectInverse>,
    Time::ThreePointWaveletFn, Time::ThreePointWaveletFn>;
extern template class BlockDiagonalBilinearForm<
    space::MultigridPreconditioner<space::StiffnessOperator>,
    Time::OrthonormalWaveletFn, Time::OrthonormalWaveletFn>;
extern template class BlockDiagonalBilinearForm<
    space::XPreconditionerOperator<space::MultigridPreconditioner>,
    Time::ThreePointWaveletFn, Time::ThreePointWaveletFn>;

}  // namespace spacetime

#include "bilinear_form.ipp"
