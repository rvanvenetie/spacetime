#pragma once
#include <utility>
#include <vector>

#include "basis.hpp"
#include "sparse_vector.hpp"

namespace Time {

template <typename basis_in, typename basis_out>
class LinearOperator {
 public:
  // This matvec applies the operator column-wise.
  SparseVector<basis_out> matvec(const SparseVector<basis_in> &vec) const;
  SparseVector<basis_in> rmatvec(const SparseVector<basis_out> &vec) const;

  // This matvec applies the operator row-wise for the given output indices.
  SparseVector<basis_out> matvec(const SparseVector<basis_in> &vec,
                                 std::vector<basis_out *> indices_out) const;
  SparseVector<basis_in> rmatvec(const SparseVector<basis_out> &vec,
                                 std::vector<basis_in *> indices_out) const;

  // Create functor operators, for convenience.
  SparseVector<basis_out> operator()(const SparseVector<basis_in> &vec) const {
    return matvec(vec);
  }
  SparseVector<basis_out> operator()(
      const SparseVector<basis_in> &vec,
      std::vector<basis_out *> indices_out) const {
    return matvec(vec, indices_out);
  }

  // Column should be the column vector associated to the given basis function.
  virtual SparseVector<basis_out> Column(basis_in *phi_in) const = 0;

  // Row should be the column vector associated to the given basis function.
  virtual SparseVector<basis_in> Row(basis_out *phi_out) const = 0;
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
template <typename basis>
class Prolongate : public LinearOperator<basis, basis> {
  SparseVector<basis> Column(basis *phi_in) const final;

  // Also named the `restriction` operator
  SparseVector<basis> Row(basis *phi_out) const final;
};

/**
 *   Below are the single scale (levelwise) operators.
 */
template <typename basis_in, typename basis_out>
class MassOperator : public LinearOperator<basis_in, basis_out> {
  SparseVector<basis_out> Column(basis_in *phi_in) const final;
  SparseVector<basis_in> Row(basis_out *phi_out) const final;
};

// Self-adjoint operator.

}  // namespace Time

#include "linear_operator.ipp"
