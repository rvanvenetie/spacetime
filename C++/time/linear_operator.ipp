#pragma once

#include <boost/container/static_vector.hpp>
#include <iostream>
#include <unordered_map>

#include "haar_basis.hpp"
#include "linear_operator.hpp"
#include "orthonormal_basis.hpp"
#include "three_point_basis.hpp"
namespace Time {

/**
 *  Implementations of LinearOperator.
 */
template <typename I, typename BasisIn, typename BasisOut>
SparseVector<BasisOut> LinearOperator<I, BasisIn, BasisOut>::MatVec(
    const SparseVector<BasisIn> &vec) const {
  SparseVector<BasisOut> result;
  result.reserve(vec.size() * 2);

  for (const auto [labda_in, coeff_in] : vec)
    for (const auto [labda_out, coeff_out] : I::Column(labda_in))
      result.emplace_back(labda_out, coeff_in * coeff_out);
  result.Compress();
  return result;
}

template <typename I, typename BasisIn, typename BasisOut>
SparseVector<BasisIn> LinearOperator<I, BasisIn, BasisOut>::RMatVec(
    const SparseVector<BasisOut> &vec) const {
  SparseVector<BasisIn> result;
  result.reserve(vec.size() * 2);

  for (const auto [labda_out, coeff_out] : vec)
    for (const auto [labda_in, coeff_in] : I::Row(labda_out))
      result.emplace_back(labda_in, coeff_out * coeff_in);
  result.Compress();
  return result;
}

template <typename I, typename BasisIn, typename BasisOut>
SparseVector<BasisOut> LinearOperator<I, BasisIn, BasisOut>::MatVec(
    const SparseVector<BasisIn> &vec,
    const SparseIndices<BasisOut> &indices_out) const {
  if (vec.empty() || indices_out.empty()) return {};
  assert(indices_out.IsUnique());
  SparseVector<BasisOut> result;
  result.reserve(indices_out.size());

  vec.StoreInTree();
  for (const auto labda_out : indices_out) {
    double val = 0;
    for (const auto [labda_in, coeff_in] : I::Row(labda_out))
      if (labda_in->has_data())
        val += coeff_in * (*labda_in->template data<double>());
    result.emplace_back(labda_out, val);
  }
  vec.RemoveFromTree();
  return result;
}

template <typename I, typename BasisIn, typename BasisOut>
SparseVector<BasisIn> LinearOperator<I, BasisIn, BasisOut>::RMatVec(
    const SparseVector<BasisOut> &vec,
    const SparseIndices<BasisIn> &indices_in) const {
  if (vec.empty() || indices_in.empty()) return {};
  assert(indices_in.IsUnique());
  SparseVector<BasisIn> result;
  result.reserve(indices_in.size());

  vec.StoreInTree();
  for (const auto labda_in : indices_in) {
    double val = 0;
    for (const auto [labda_out, coeff_out] : I::Column(labda_in))
      if (labda_out->has_data())
        val += coeff_out * (*labda_out->template data<double>());
    result.emplace_back(labda_in, val);
  }
  vec.RemoveFromTree();
  return result;
}

template <typename I, typename BasisIn, typename BasisOut>
SparseIndices<BasisOut> LinearOperator<I, BasisIn, BasisOut>::Range(
    const SparseIndices<BasisIn> &ind) const {
  SparseIndices<BasisOut> result;
  result.reserve(ind.size() * 2);

  for (const auto labda_in : ind)
    for (const auto [labda_out, _] : I::Column(labda_in))
      result.emplace_back(labda_out);
  result.Compress();
  return result;
}

template <typename I, typename BasisIn, typename BasisOut>
Eigen::MatrixXd LinearOperator<I, BasisIn, BasisOut>::ToMatrix(
    const SparseIndices<BasisIn> &indices_in,
    const SparseIndices<BasisOut> &indices_out) const {
  assert(indices_in.IsUnique() && indices_out.IsUnique());
  std::unordered_map<BasisIn *, int> indices_in_map;
  std::unordered_map<BasisOut *, int> indices_out_map;
  Eigen::MatrixXd A =
      Eigen::MatrixXd::Zero(indices_out.size(), indices_in.size());

  for (int i = 0; i < indices_in.size(); ++i) {
    assert(!indices_in_map.count(indices_in[i]));
    indices_in_map[indices_in[i]] = i;
  }
  for (int i = 0; i < indices_out.size(); ++i) {
    assert(!indices_out_map.count(indices_out[i]));
    indices_out_map[indices_out[i]] = i;
  }

  // Create A.
  for (int i = 0; i < indices_in.size(); ++i) {
    SparseVector<BasisIn> vec{{{indices_in[i], 1.0}}};
    auto op_vec = MatVec(vec);
    for (auto [fn, coeff] : op_vec) {
      A(indices_out_map[fn], i) = coeff;
    }
  }

  return A;
}

/**
 *  Below implementations have various return types based. The aliases
 *  below are some handy containers with fixed memory size.
 */
template <typename Basis, size_t N>
using StaticSparseVector =
    boost::container::static_vector<std::pair<Basis *, double>, N>;
template <typename Basis, size_t N>
using ArraySparseVector = std::array<std::pair<Basis *, double>, N>;

/**
 *  Implementations of Prolongate.
 */
template <>
inline auto Prolongate<ContLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) {
  StaticSparseVector<ContLinearScalingFn, 3> result;

  auto [l, n] = phi_in->labda();
  result.emplace_back(phi_in->RefineMiddle(), 1.0);
  if (n > 0) result.emplace_back(phi_in->RefineLeft(), 0.5);
  if (n < (1 << l)) result.emplace_back(phi_in->RefineRight(), 0.5);
  return result;
}

template <>
inline auto Prolongate<ContLinearScalingFn>::Row(ContLinearScalingFn *phi_in) {
  StaticSparseVector<ContLinearScalingFn, 2> result;
  const auto &parents = phi_in->parents();
  if (parents.size() == 1)
    result = {{{parents[0], 1.0}}};
  else if (parents.size() == 2)
    result = {{{parents[0], 0.5}, {parents[1], 0.5}}};
  else
    assert(false);
  return result;
}

template <>
inline auto Prolongate<DiscLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) {
  StaticSparseVector<DiscLinearScalingFn, 4> result;
  auto [l, n] = phi_in->labda();
  phi_in->Refine();
  const auto &children = phi_in->children();
  if (phi_in->pw_constant()) {
    result.emplace_back(children[0], 1);
    result.emplace_back(children[2], 1);
  } else {
    result.emplace_back(children[0], -sqrt(3) / 2);
    result.emplace_back(children[1], 0.5);
    result.emplace_back(children[2], sqrt(3) / 2);
    result.emplace_back(children[3], 0.5);
  }
  return result;
}

template <>
inline auto Prolongate<DiscLinearScalingFn>::Row(DiscLinearScalingFn *phi_in) {
  StaticSparseVector<DiscLinearScalingFn, 2> result;
  auto [l, n] = phi_in->labda();
  const auto &parents = phi_in->parents();
  switch (n % 4) {
    case 0:
      result = {{{parents[0], 1.0}, {parents[1], -sqrt(3) / 2}}};
      return result;
    case 1:
      result = {{{parents[1], 0.5}}};
      return result;
    case 2:
      result = {{{parents[0], 1.0}, {parents[1], sqrt(3) / 2}}};
      return result;
    case 3:
    default:
      result = {{{parents[1], 0.5}}};
      return result;
  }
}

/**
 *  Implementations of the Mass operator.
 */
template <>
inline auto MassOperator<ContLinearScalingFn, ContLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) {
  StaticSparseVector<ContLinearScalingFn, 3> result;

  auto [l, n] = phi_in->labda();
  double self_ip = 0;
  if (n > 0) {
    auto elem = phi_in->support()[0];
    result.emplace_back(elem->RefineContLinear()[0], 1. / ((1 << l) * 6.0));
    self_ip += 1. / ((1 << l) * 3.0);
  }
  if (n < (1 << l)) {
    auto elem = phi_in->support().back();
    result.emplace_back(elem->RefineContLinear()[1], 1. / ((1 << l) * 6.0));
    self_ip += 1. / ((1 << l) * 3.0);
  }
  result.emplace_back(phi_in, self_ip);
  return result;
}

template <>
inline auto MassOperator<ContLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) {
  // This MassOperator is symmetric.
  return Column(phi_out);
}

template <>
inline auto MassOperator<DiscConstantScalingFn, DiscConstantScalingFn>::Column(
    DiscConstantScalingFn *phi_in) {
  return ArraySparseVector<DiscConstantScalingFn, 1>{
      {{phi_in, 1. / (1 << phi_in->level())}}};
}

template <>
inline auto MassOperator<DiscConstantScalingFn, DiscConstantScalingFn>::Row(
    DiscConstantScalingFn *phi_out) {
  // This MassOperator is symmetric.
  return Column(phi_out);
}

template <>
inline auto MassOperator<DiscLinearScalingFn, DiscLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) {
  return ArraySparseVector<DiscLinearScalingFn, 1>{
      {{phi_in, 1. / (1 << phi_in->level())}}};
}

template <>
inline auto MassOperator<DiscLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) {
  // This MassOperator is symmetric.
  return Column(phi_out);
}

namespace Mass {
inline auto ThreeInOrthoOut(ContLinearScalingFn *phi_in) {
  StaticSparseVector<DiscLinearScalingFn, 4> result;

  auto [l, n] = phi_in->labda();
  const double s = 1. / (1 << (l + 1));  // s = pow(2, -(l+1)).
  if (n > 0) {
    const auto &[pdl0, pdl1] = phi_in->support().front()->PhiDiscLinear();
    assert(pdl0 != nullptr && pdl1 != nullptr);
    result.emplace_back(pdl0, s);
    result.emplace_back(pdl1, s / sqrt(3));
  }
  if (n < (1 << l)) {
    const auto &[pdl0, pdl1] = phi_in->support().back()->PhiDiscLinear();
    assert(pdl0 != nullptr && pdl1 != nullptr);
    result.emplace_back(pdl0, s);
    result.emplace_back(pdl1, -s / sqrt(3));
  }
  return result;
}

inline auto OrthoInThreeOut(DiscLinearScalingFn *phi_in) {
  ArraySparseVector<ContLinearScalingFn, 2> result;
  auto [l, n] = phi_in->labda();
  const auto &[pcl0, pcl1] = phi_in->support()[0]->RefineContLinear();
  assert(pcl0 != nullptr && pcl1 != nullptr);
  const double s = 1. / (1 << (l + 1));  // s = pow(2, -(l+1)).
  if (phi_in->pw_constant())
    result = {{{pcl0, s}, {pcl1, s}}};
  else
    result = {{{pcl0, -s / sqrt(3)}, {pcl1, s / sqrt(3)}}};
  return result;
}
};  // namespace Mass

template <>
inline auto MassOperator<ContLinearScalingFn, DiscLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) {
  return Mass::ThreeInOrthoOut(phi_in);
}

template <>
inline auto MassOperator<ContLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) {
  return Mass::OrthoInThreeOut(phi_out);
}

template <>
inline auto MassOperator<DiscLinearScalingFn, ContLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) {
  return Mass::OrthoInThreeOut(phi_in);
}

template <>
inline auto MassOperator<DiscLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) {
  return Mass::ThreeInOrthoOut(phi_out);
}

template <>
inline auto ZeroEvalOperator<ContLinearScalingFn, ContLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) {
  StaticSparseVector<ContLinearScalingFn, 1> result;
  auto [l, n] = phi_in->labda();
  if (n == 0) result = {{{phi_in, 1.0}}};
  return result;
}

template <>
inline auto ZeroEvalOperator<ContLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) {
  return Column(phi_out);
}

template <>
inline auto ZeroEvalOperator<DiscLinearScalingFn, DiscLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) {
  StaticSparseVector<DiscLinearScalingFn, 2> result;
  auto [l, n] = phi_in->labda();
  if (n <= 1) {
    const auto &[pdl0, pdl1] = phi_in->support().front()->PhiDiscLinear();
    assert(pdl0 != nullptr && pdl1 != nullptr);
    if (phi_in->pw_constant())
      result = {{{pdl0, 1.0}, {pdl1, -sqrt(3)}}};
    else
      result = {{{pdl0, -sqrt(3)}, {pdl1, 3.0}}};
  }
  return result;
}

template <>
inline auto ZeroEvalOperator<DiscLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) {
  return Column(phi_out);
}

namespace ZeroEval {
inline auto ThreeInOrthoOut(ContLinearScalingFn *phi_in) {
  StaticSparseVector<DiscLinearScalingFn, 2> result;
  auto [l, n] = phi_in->labda();
  if (n == 0) {
    const auto &[pdl0, pdl1] = phi_in->support().front()->PhiDiscLinear();
    assert(pdl0 != nullptr && pdl1 != nullptr);
    result = {{{pdl0, 1.0}, {pdl1, -sqrt(3)}}};
  }
  return result;
}

inline auto OrthoInThreeOut(DiscLinearScalingFn *phi_in) {
  StaticSparseVector<ContLinearScalingFn, 1> result;
  auto [l, n] = phi_in->labda();
  if (n <= 1) {
    const auto &pcl0 = phi_in->support().front()->RefineContLinear()[0];
    assert(pcl0 != nullptr);
    if (phi_in->pw_constant())
      result = {{{pcl0, 1.0}}};
    else
      result = {{{pcl0, -sqrt(3)}}};
  }
  return result;
}
};  // namespace ZeroEval

template <>
inline auto ZeroEvalOperator<ContLinearScalingFn, DiscLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) {
  return ZeroEval::ThreeInOrthoOut(phi_in);
}

template <>
inline auto ZeroEvalOperator<ContLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) {
  return ZeroEval::OrthoInThreeOut(phi_out);
}

template <>
inline auto ZeroEvalOperator<DiscLinearScalingFn, ContLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) {
  return ZeroEval::OrthoInThreeOut(phi_in);
}

template <>
inline auto ZeroEvalOperator<DiscLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) {
  return ZeroEval::ThreeInOrthoOut(phi_out);
}

namespace Transport {
inline auto ThreeInOrthoOut(ContLinearScalingFn *phi_in) {
  StaticSparseVector<DiscLinearScalingFn, 2> result;

  auto [l, n] = phi_in->labda();
  if (n > 0) {
    const auto &pdl0 = phi_in->support().front()->PhiDiscLinear()[0];
    assert(pdl0 != nullptr);
    result.emplace_back(pdl0, 1.0);
  }
  if (n < (1 << l)) {
    const auto &pdl0 = phi_in->support().back()->PhiDiscLinear()[0];
    assert(pdl0 != nullptr);
    result.emplace_back(pdl0, -1.0);
  }
  return result;
}
inline auto OrthoInThreeOut(DiscLinearScalingFn *phi_in) {
  StaticSparseVector<ContLinearScalingFn, 2> result;
  auto [l, n] = phi_in->labda();
  const auto &[pcl0, pcl1] = phi_in->support().front()->RefineContLinear();
  assert(pcl0 != nullptr && pcl1 != nullptr);
  if (phi_in->pw_constant()) result = {{{pcl0, -1.0}, {pcl1, 1.0}}};
  return result;
}
}  // namespace Transport

template <>
inline auto TransportOperator<ContLinearScalingFn, DiscLinearScalingFn>::Column(
    ContLinearScalingFn *phi_in) {
  return Transport::ThreeInOrthoOut(phi_in);
}

template <>
inline auto TransportOperator<ContLinearScalingFn, DiscLinearScalingFn>::Row(
    DiscLinearScalingFn *phi_out) {
  return Transport::OrthoInThreeOut(phi_out);
}

template <>
inline auto TransportOperator<DiscLinearScalingFn, ContLinearScalingFn>::Column(
    DiscLinearScalingFn *phi_in) {
  return Transport::OrthoInThreeOut(phi_in);
}

template <>
inline auto TransportOperator<DiscLinearScalingFn, ContLinearScalingFn>::Row(
    ContLinearScalingFn *phi_out) {
  return Transport::ThreeInOrthoOut(phi_out);
}

}  // namespace Time
