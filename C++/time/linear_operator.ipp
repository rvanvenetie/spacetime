#pragma once
#include <iostream>

#include "haar_basis.hpp"
#include "linear_operator.hpp"
#include "three_point_basis.hpp"

namespace Time {

/**
 *  Implementations of LinearOperator.
 */
template <typename BasisIn, typename BasisOut>
SparseVector<BasisOut> LinearOperator<BasisIn, BasisOut>::MatVec(
    const SparseVector<BasisIn> &vec) const {
  static SparseVector<BasisOut> result;
  result.clear();

  for (auto [labda_in, coeff_in] : vec)
    for (auto [labda_out, coeff_out] : Column(labda_in))
      result.emplace_back(labda_out, coeff_in * coeff_out);
  result.Compress();
  return result;
}

template <typename BasisIn, typename BasisOut>
SparseVector<BasisIn> LinearOperator<BasisIn, BasisOut>::RMatVec(
    const SparseVector<BasisOut> &vec) const {
  static SparseVector<BasisIn> result;
  result.clear();

  for (auto [labda_out, coeff_out] : vec)
    for (auto [labda_in, coeff_in] : Row(labda_out))
      result.emplace_back(labda_in, coeff_out * coeff_in);
  result.Compress();
  return result;
}

template <typename BasisIn, typename BasisOut>
SparseVector<BasisOut> LinearOperator<BasisIn, BasisOut>::MatVec(
    const SparseVector<BasisIn> &vec,
    const SparseIndices<BasisOut> &indices_out) const {
  assert(indices_out.IsUnique());
  static SparseVector<BasisOut> result;
  result.clear();

  vec.StoreInTree();
  for (auto labda_out : indices_out) {
    double val = 0;
    for (auto [labda_in, coeff_in] : Row(labda_out))
      if (labda_in->has_data())
        val += coeff_in * (*labda_in->template data<double>());
    result.emplace_back(labda_out, val);
  }
  vec.RemoveFromTree();
  return result;
}
template <typename BasisIn, typename BasisOut>
SparseVector<BasisIn> LinearOperator<BasisIn, BasisOut>::RMatVec(
    const SparseVector<BasisOut> &vec,
    const SparseIndices<BasisIn> &indices_in) const {
  assert(indices_in.IsUnique());
  static SparseVector<BasisIn> result;
  result.clear();

  vec.StoreInTree();
  for (auto labda_in : indices_in) {
    double val = 0;
    for (auto [labda_out, coeff_out] : Column(labda_in))
      if (labda_out->has_data())
        val += coeff_out * (*labda_out->template data<double>());
    result.emplace_back(labda_in, val);
  }
  vec.RemoveFromTree();
  return result;
}

template <typename BasisIn, typename BasisOut>
SparseIndices<BasisOut> LinearOperator<BasisIn, BasisOut>::Range(
    const SparseIndices<BasisIn> &ind) const {
  static SparseIndices<BasisOut> result;
  result.clear();

  for (auto labda_in : ind)
    for (auto [labda_out, _] : Column(labda_in)) result.emplace_back(labda_out);
  result.Compress();
  return result;
}

/**
 *  Implementations of Prolongate.
 */

template <>
SparseVector<ContLinearScalingFn> Prolongate<ContLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) const {
  static SparseVector<ContLinearScalingFn> result;
  result.clear();

  auto [l, n] = phi_in->labda();
  result.emplace_back(phi_in->RefineMiddle(), 1.0);
  if (n > 0) result.emplace_back(phi_in->RefineLeft(), 0.5);
  if (n < (1 << l)) result.emplace_back(phi_in->RefineRight(), 0.5);
  return result;
}

template <>
SparseVector<ContLinearScalingFn> Prolongate<ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_in) const {
  static SparseVector<ContLinearScalingFn> result;
  result.clear();

  if (phi_in->parents().size() == 1) {
    result.emplace_back(phi_in->parents()[0], 1);
  } else if (phi_in->parents().size() == 2) {
    result.emplace_back(phi_in->parents()[0], 0.5);
    result.emplace_back(phi_in->parents()[1], 0.5);
  } else
    assert(false);
  return result;
}

/**
 *  Implementations of the Mass operator.
 */
template <>
SparseVector<ContLinearScalingFn>
MassOperator<ContLinearScalingFn, ContLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) const {
  static SparseVector<ContLinearScalingFn> result;
  result.clear();

  auto [l, n] = phi_in->labda();
  double self_ip = 0;
  if (n > 0) {
    auto elem = phi_in->support()[0];
    result.emplace_back(elem->RefineContLinear()[0], 1. / 6 * pow(2, -l));
    self_ip += 1. / 3 * pow(2, -l);
  }
  if (n < (1 << l)) {
    auto elem = phi_in->support().back();
    result.emplace_back(elem->RefineContLinear()[1], 1. / 6 * pow(2, -l));
    self_ip += 1. / 3 * pow(2, -l);
  }
  result.emplace_back(phi_in, self_ip);
  return result;
}

template <>
SparseVector<ContLinearScalingFn>
MassOperator<ContLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) const {
  return Column(phi_out);
}

}  // namespace Time
