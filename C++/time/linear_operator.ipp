#pragma once
#include <iostream>

#include "haar_basis.hpp"
#include "linear_operator.hpp"
#include "three_point_basis.hpp"

namespace Time {

template <typename basis_in, typename basis_out>
SparseVector<basis_out> LinearOperator<basis_in, basis_out>::matvec(
    const SparseVector<basis_in> &vec) const {
  SparseVector<basis_out> result;
  for (auto [labda_in, coeff_in] : vec)
    for (auto [labda_out, coeff_out] : Column(labda_in))
      result.emplace_back(labda_out, coeff_in * coeff_out);
  result.Compress();
  return result;
}

template <typename basis_in, typename basis_out>
SparseVector<basis_in> LinearOperator<basis_in, basis_out>::rmatvec(
    const SparseVector<basis_out> &vec) const {
  SparseVector<basis_in> result;
  for (auto [labda_out, coeff_out] : vec)
    for (auto [labda_in, coeff_in] : Row(labda_out))
      result.emplace_back(labda_in, coeff_out * coeff_in);
  result.Compress();
  return result;
}

template <typename basis_in, typename basis_out>
SparseVector<basis_out> LinearOperator<basis_in, basis_out>::matvec(
    const SparseVector<basis_in> &vec,
    std::vector<basis_out *> indices_out) const {
  vec.StoreInTree();
  SparseVector<basis_out> result;
  for (auto labda_out : indices_out)
    for (auto [labda_in, coeff_in] : Row(labda_out))
      if (labda_in->has_data())
        result.emplace_back(labda_out,
                            coeff_in * (*labda_in->template data<double>()));
  vec.RemoveFromTree();
  result.Compress();
  return result;
}
template <typename basis_in, typename basis_out>
SparseVector<basis_in> LinearOperator<basis_in, basis_out>::rmatvec(
    const SparseVector<basis_out> &vec,
    std::vector<basis_in *> indices_in) const {
  vec.StoreInTree();
  SparseVector<basis_in> result;
  for (auto labda_in : indices_in)
    for (auto [labda_out, coeff_out] : Column(labda_in))
      if (labda_out->has_data())
        result.emplace_back(labda_in,
                            coeff_out * (*labda_out->template data<double>()));
  vec.RemoveFromTree();
  result.Compress();
  return result;
}

template <>
SparseVector<ContLinearScalingFn> Prolongate<ContLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) const {
  auto [l, n] = phi_in->labda();
  SparseVector<ContLinearScalingFn> result;
  result.emplace_back(phi_in->RefineMiddle(), 1.0);
  if (n > 0) result.emplace_back(phi_in->RefineLeft(), 0.5);
  if (n < (1 << l)) result.emplace_back(phi_in->RefineRight(), 0.5);
  return result;
}

template <>
SparseVector<ContLinearScalingFn> Prolongate<ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_in) const {
  SparseVector<ContLinearScalingFn> result;
  if (phi_in->parents().size() == 1) {
    result.emplace_back(phi_in->parents()[0], 1);
  } else if (phi_in->parents().size() == 2) {
    result.emplace_back(phi_in->parents()[0], 0.5);
    result.emplace_back(phi_in->parents()[1], 0.5);
  } else
    assert(false);
  return result;
}

template <>
SparseVector<ContLinearScalingFn>
MassOperator<ContLinearScalingFn, ContLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) const {
  SparseVector<ContLinearScalingFn> result;
  auto [l, n] = phi_in->labda();
  double self_ip = 0;
  if (n > 0) {
    auto elem = phi_in->support()[0];
    result.emplace_back(elem->RefineContLinear()[0], 1 / 6 * pow(2, -l));
    self_ip += 1 / 3 * pow(2, -l);
  }
  if (n < (1 << l)) {
    auto elem = phi_in->support().back();
    result.emplace_back(elem->RefineContLinear()[1], 1 / 6 * pow(2, -l));
    self_ip += 1 / 3 * pow(2, -l);
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
