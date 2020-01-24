#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>

#include "basis.hpp"
#include "sparse_vector.hpp"

namespace Time {

template <typename BasisIn, typename BasisOut>
class LinearOperator {
 public:
  // This MatVec applies the operator column-wise.
  SparseVector<BasisOut> MatVec(const SparseVector<BasisIn> &vec) const;
  SparseVector<BasisIn> RMatVec(const SparseVector<BasisOut> &vec) const;

  // This MatVec applies the operator row-wise for the given output indices.
  SparseVector<BasisOut> MatVec(
      const SparseVector<BasisIn> &vec,
      const SparseIndices<BasisOut> &indices_out) const;
  SparseVector<BasisIn> RMatVec(const SparseVector<BasisOut> &vec,
                                const SparseIndices<BasisIn> &indices_in) const;

  // Create functor operators, for convenience.
  SparseVector<BasisOut> operator()(const SparseVector<BasisIn> &vec) const {
    return MatVec(vec);
  }
  SparseVector<BasisOut> operator()(
      const SparseVector<BasisIn> &vec,
      const SparseIndices<BasisOut> &indices_out) const {
    return MatVec(vec, indices_out);
  }

  // Return the range of this operator if you were to apply the given indices.
  SparseIndices<BasisOut> Range(const SparseIndices<BasisIn> &ind) const;

  // Column should be the column vector associated to the given basis function.
  virtual SparseVector<BasisOut> Column(BasisIn *phi_in) const = 0;

  // Row should be the column vector associated to the given basis function.
  virtual SparseVector<BasisIn> Row(BasisOut *phi_out) const = 0;

  // Debug function, O(n^2).
  Eigen::MatrixXd ToMatrix(const SparseIndices<BasisIn> &indices_in,
                           const SparseIndices<BasisOut> &indices_out) const;
};

/**
 *  Below are the operations for transformations between single and multi scale.
 */
template <typename Wavelet>
class WaveletToScaling
    : public LinearOperator<Wavelet, typename FunctionTrait<Wavelet>::Scaling> {
  SparseVector<typename FunctionTrait<Wavelet>::Scaling> Column(
      Wavelet *psi_in) const final {
    return psi_in->single_scale();
  }

  SparseVector<Wavelet> Row(
      typename FunctionTrait<Wavelet>::Scaling *phi_out) const final {
    return phi_out->multi_scale();
  }
};

/**
 *  Below are the prolongation/restriction operators for single scale functions.
 */
template <typename Basis>
class Prolongate : public LinearOperator<Basis, Basis> {
  SparseVector<Basis> Column(Basis *phi_in) const final;

  // Also named the `restriction` operator
  SparseVector<Basis> Row(Basis *phi_out) const final;
};

/**
 *   Below are the single scale (levelwise) operators.
 */
template <typename BasisIn, typename BasisOut>
class MassOperator : public LinearOperator<BasisIn, BasisOut> {
  SparseVector<BasisOut> Column(BasisIn *phi_in) const final;
  SparseVector<BasisIn> Row(BasisOut *phi_out) const final;
};

// Evaluates the functions in zero: <gamma_0 phi, gamma_0 psi> = phi(0) psi(0).
template <typename BasisIn, typename BasisOut>
class ZeroEvalOperator : public LinearOperator<BasisIn, BasisOut> {
  SparseVector<BasisOut> Column(BasisIn *phi_in) const final;
  SparseVector<BasisIn> Row(BasisOut *phi_out) const final;
};

// Transport matrix <phi, d/dt psi>.
template <typename BasisIn, typename BasisOut>
class TransportOperator : public LinearOperator<BasisIn, BasisOut> {
  SparseVector<BasisOut> Column(BasisIn *phi_in) const final;
  SparseVector<BasisIn> Row(BasisOut *phi_out) const final;
};

}  // namespace Time

#include "linear_operator.ipp"
