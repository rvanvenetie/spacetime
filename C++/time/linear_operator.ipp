#pragma once
#include <iostream>

#include "haar_basis.hpp"
#include "linear_operator.hpp"
#include "orthonormal_basis.hpp"
#include "three_point_basis.hpp"

namespace Time {

/**
 *  Implementations of LinearOperator.
 */
template <typename basis_in, typename basis_out>
SparseVector<basis_out> LinearOperator<basis_in, basis_out>::MatVec(
    const SparseVector<basis_in> &vec) const {
  SparseVector<basis_out> result;
  for (auto [labda_in, coeff_in] : vec)
    for (auto [labda_out, coeff_out] : Column(labda_in))
      result.emplace_back(labda_out, coeff_in * coeff_out);
  result.Compress();
  return result;
}

template <typename basis_in, typename basis_out>
SparseVector<basis_in> LinearOperator<basis_in, basis_out>::RMatVec(
    const SparseVector<basis_out> &vec) const {
  SparseVector<basis_in> result;
  for (auto [labda_out, coeff_out] : vec)
    for (auto [labda_in, coeff_in] : Row(labda_out))
      result.emplace_back(labda_in, coeff_out * coeff_in);
  result.Compress();
  return result;
}

template <typename basis_in, typename basis_out>
SparseVector<basis_out> LinearOperator<basis_in, basis_out>::MatVec(
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
SparseVector<basis_in> LinearOperator<basis_in, basis_out>::RMatVec(
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

/**
 *  Implementations of Prolongate.
 */

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
  if (phi_in->parents().size() == 1)
    return {{{phi_in->parents()[0], 1.0}}};
  else if (phi_in->parents().size() == 2)
    return {{{phi_in->parents()[0], 0.5}, {phi_in->parents()[1], 0.5}}};
  else
    assert(false);
}

template <>
SparseVector<DiscLinearScalingFn> Prolongate<DiscLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) const {
  auto [l, n] = phi_in->labda();
  phi_in->Refine();
  auto children = phi_in->children();
  if (phi_in->pw_constant())
    return {{{children[0].get(), 1.0}, {children[2].get(), 1.0}}};
  else
    return {{{children[0].get(), -sqrt(3) / 2},
             {children[1].get(), 0.5},
             {children[2].get(), sqrt(3) / 2},
             {children[3].get(), 0.5}}};
}

template <>
SparseVector<DiscLinearScalingFn> Prolongate<DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_in) const {
  auto [l, n] = phi_in->labda();
  auto parents = phi_in->parents();
  switch (n % 4) {
    case 0:
      return {{{parents[0], 1.0}, {parents[1], -sqrt(3) / 2}}};
    case 1:
      return {{{parents[1], 0.5}}};
    case 2:
      return {{{parents[0], 1.0}, {parents[1], sqrt(3) / 2}}};
    case 3:
    default:
      return {{{parents[1], 0.5}}};
  }
}

/**
 *  Implementations of the Mass operator.
 */
template <>
SparseVector<ContLinearScalingFn>
MassOperator<ContLinearScalingFn, ContLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) const {
  SparseVector<ContLinearScalingFn> result;
  auto [l, n] = phi_in->labda();
  double self_ip = 0;
  if (n > 0) {
    auto elem = phi_in->support()[0];
    result.emplace_back(elem->RefineContLinear()[0], pow(2, -l) / 6.0);
    self_ip += pow(2, -l) / 3.0;
  }
  if (n < (1 << l)) {
    auto elem = phi_in->support().back();
    result.emplace_back(elem->RefineContLinear()[1], pow(2, -l) / 6.0);
    self_ip += pow(2, -l) / 3.0;
  }
  result.emplace_back(phi_in, self_ip);
  return result;
}

template <>
SparseVector<ContLinearScalingFn>
MassOperator<ContLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) const {
  // This MassOperator is symmetric.
  return Column(phi_out);
}

template <>
SparseVector<DiscConstantScalingFn>
MassOperator<DiscConstantScalingFn, DiscConstantScalingFn>::Column(
    DiscConstantScalingFn *phi_in) const {
  return {{{phi_in, pow(2.0, -phi_in->level())}}};
}

template <>
SparseVector<DiscConstantScalingFn>
MassOperator<DiscConstantScalingFn, DiscConstantScalingFn>::Row(
    DiscConstantScalingFn *phi_out) const {
  // This MassOperator is symmetric.
  return Column(phi_out);
}

template <>
SparseVector<DiscLinearScalingFn>
MassOperator<DiscLinearScalingFn, DiscLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) const {
  return {{{phi_in, pow(2.0, -phi_in->level())}}};
}

template <>
SparseVector<DiscLinearScalingFn>
MassOperator<DiscLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) const {
  // This MassOperator is symmetric.
  return Column(phi_out);
}

SparseVector<DiscLinearScalingFn> mass_three_in_ortho_out(
    ContLinearScalingFn *phi_in) {
  SparseVector<DiscLinearScalingFn> result;
  auto [l, n] = phi_in->labda();
  if (n > 0) {
    auto [pdl0, pdl1] = phi_in->support().front()->PhiDiscLinear();
    assert(pdl0 != nullptr && pdl1 != nullptr);
    result.emplace_back(pdl0, pow(2.0, -(l + 1)));
    result.emplace_back(pdl1, pow(2.0, -(l + 1)) / sqrt(3));
  }
  if (n < (1 << l)) {
    auto [pdl0, pdl1] = phi_in->support().back()->PhiDiscLinear();
    assert(pdl0 != nullptr && pdl1 != nullptr);
    result.emplace_back(pdl0, pow(2.0, -(l + 1)));
    result.emplace_back(pdl1, -pow(2.0, -(l + 1)) / sqrt(3));
  }
  return result;
}

SparseVector<ContLinearScalingFn> mass_ortho_in_three_out(
    DiscLinearScalingFn *phi_in) {
  auto [l, n] = phi_in->labda();
  auto [pcl0, pcl1] = phi_in->support()[0]->RefineContLinear();
  assert(pcl0 != nullptr && pcl1 != nullptr);
  if (phi_in->pw_constant())
    return {{{pcl0, pow(2.0, -(l + 1))}, {pcl1, pow(2.0, -(l + 1))}}};
  else
    return {{{pcl0, -pow(2.0, -(l + 1)) / sqrt(3)},
             {pcl1, pow(2.0, -(l + 1)) / sqrt(3)}}};
}

template <>
SparseVector<DiscLinearScalingFn>
MassOperator<ContLinearScalingFn, DiscLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) const {
  return mass_three_in_ortho_out(phi_in);
}

template <>
SparseVector<ContLinearScalingFn>
MassOperator<ContLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) const {
  return mass_ortho_in_three_out(phi_out);
}

template <>
SparseVector<ContLinearScalingFn>
MassOperator<DiscLinearScalingFn, ContLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) const {
  return mass_ortho_in_three_out(phi_in);
}

template <>
SparseVector<DiscLinearScalingFn>
MassOperator<DiscLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) const {
  return mass_three_in_ortho_out(phi_out);
}

template <>
SparseVector<ContLinearScalingFn>
ZeroEvalOperator<ContLinearScalingFn, ContLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) const {
  auto [l, n] = phi_in->labda();
  if (n > 0) return {};
  return {{{phi_in, 1.0}}};
}

template <>
SparseVector<ContLinearScalingFn>
ZeroEvalOperator<ContLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) const {
  return Column(phi_out);
}

template <>
SparseVector<DiscLinearScalingFn>
ZeroEvalOperator<DiscLinearScalingFn, DiscLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) const {
  auto [l, n] = phi_in->labda();
  if (n > 1) return {};
  auto [pdl0, pdl1] = phi_in->support().front()->PhiDiscLinear();
  assert(pdl0 != nullptr && pdl1 != nullptr);
  if (phi_in->pw_constant())
    return {{{pdl0, 1.0}, {pdl1, -sqrt(3)}}};
  else
    return {{{pdl0, -sqrt(3)}, {pdl1, 3.0}}};
}

template <>
SparseVector<DiscLinearScalingFn>
ZeroEvalOperator<DiscLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) const {
  return Column(phi_out);
}

SparseVector<DiscLinearScalingFn> zero_eval_three_in_ortho_out(
    ContLinearScalingFn *phi_in) {
  auto [l, n] = phi_in->labda();
  if (n > 0) return {};
  auto [pdl0, pdl1] = phi_in->support().front()->PhiDiscLinear();
  assert(pdl0 != nullptr && pdl1 != nullptr);
  return {{{pdl0, 1.0}, {pdl1, -sqrt(3)}}};
}

SparseVector<ContLinearScalingFn> zero_eval_ortho_in_three_out(
    DiscLinearScalingFn *phi_in) {
  auto [l, n] = phi_in->labda();
  if (n > 1) return {};
  auto [pcl0, _] = phi_in->support().front()->RefineContLinear();
  assert(pcl0 != nullptr);
  if (phi_in->pw_constant())
    return {{{pcl0, 1.0}}};
  else
    return {{{pcl0, -sqrt(3)}}};
}

template <>
SparseVector<DiscLinearScalingFn>
ZeroEvalOperator<ContLinearScalingFn, DiscLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) const {
  return zero_eval_three_in_ortho_out(phi_in);
}

template <>
SparseVector<ContLinearScalingFn>
ZeroEvalOperator<ContLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) const {
  return zero_eval_ortho_in_three_out(phi_out);
}

template <>
SparseVector<ContLinearScalingFn>
ZeroEvalOperator<DiscLinearScalingFn, ContLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) const {
  return zero_eval_ortho_in_three_out(phi_in);
}

template <>
SparseVector<DiscLinearScalingFn>
ZeroEvalOperator<DiscLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) const {
  return zero_eval_three_in_ortho_out(phi_out);
}

template <>
SparseVector<DiscLinearScalingFn>
TransportOperator<ContLinearScalingFn, DiscLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) const {
  SparseVector<DiscLinearScalingFn> result;
  auto [l, n] = phi_in->labda();
  if (n > 0) {
    auto [pdl0, _] = phi_in->support().front()->PhiDiscLinear();
    assert(pdl0 != nullptr);
    result.emplace_back(pdl0, 1.0);
  }
  if (n < (1 << l)) {
    auto [pdl0, _] = phi_in->support().back()->PhiDiscLinear();
    assert(pdl0 != nullptr);
    result.emplace_back(pdl0, -1.0);
  }
  return result;
}

template <>
SparseVector<ContLinearScalingFn>
TransportOperator<ContLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) const {
  auto [l, n] = phi_out->labda();
  auto [pcl0, pcl1] = phi_out->support().front()->RefineContLinear();
  assert(pcl0 != nullptr && pcl1 != nullptr);
  if (phi_out->pw_constant())
    return {{{pcl0, -1.0}, {pcl1, 1.0}}};
  else
    return {};
}

}  // namespace Time
