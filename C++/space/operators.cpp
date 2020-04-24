#define EIGEN_NO_DEBUG
#include "operators.hpp"

#include <vector>

using Eigen::VectorXd;

namespace space {

bool Operator::FeasibleVector(const Eigen::VectorXd &vec) const {
  for (int i = 0; i < triang_.V; ++i)
    if (!IsDof(i) && vec[i] != 0) return false;

  return true;
}

Eigen::MatrixXd Operator::ToMatrix() const {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(triang_.V, triang_.V);
  for (int i = 0; i < triang_.V; ++i) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(triang_.V);
    if (IsDof(i)) v[i] = 1;
    Apply(v);
    A.col(i) = v;
  }
  return A;
}

template <class ForwardOp>
ForwardOperator<ForwardOp>::ForwardOperator(const TriangulationView &triang,
                                            OperatorOptions opts)
    : Operator(triang, opts), matrix_(ComputeMatrixSingleScale()) {}

template <class ForwardOp>
void ForwardOperator<ForwardOp>::Apply(Eigen::VectorXd &v) const {
  assert(FeasibleVector(v));

  // Hierarhical basis to single scale basis.
  ApplyHierarchToSingle(v);
  assert(FeasibleVector(v));

  // Apply the operator in single scale.
  ApplySingleScale(v);
  assert(FeasibleVector(v));

  // Return back to hierarhical basis.
  ApplyTransposeHierarchToSingle(v);
  assert(FeasibleVector(v));
}

template <class ForwardOp>
void ForwardOperator<ForwardOp>::ApplyHierarchToSingle(VectorXd &w) const {
  for (size_t vi = triang_.InitialVertices(); vi < triang_.V; ++vi)
    for (auto gp : triang_.Godparents(vi)) w[vi] = w[vi] + 0.5 * w[gp];
}

template <class ForwardOp>
void ForwardOperator<ForwardOp>::ApplyTransposeHierarchToSingle(
    VectorXd &w) const {
  for (size_t vi = triang_.V - 1; vi >= triang_.InitialVertices(); --vi)
    for (auto gp : triang_.Godparents(vi))
      if (IsDof(gp)) w[gp] = w[gp] + 0.5 * w[vi];
}

template <class ForwardOp>
void ForwardOperator<ForwardOp>::ApplySingleScale(Eigen::VectorXd &v) const {
  if (opts_.cache_forward_mat_) v = MatrixSingleScale() * v;

  auto &vertices = triang_.vertices();
  Eigen::VectorXd result = Eigen::VectorXd::Zero(v.rows());
  for (const auto &elem : triang_.elements()) {
    if (!elem->is_leaf()) continue;
    auto &Vids = elem->vertices_view_idx_;
    auto &&element_mat = ForwardOp::ElementMatrix(elem, opts_);

    for (size_t i = 0; i < 3; ++i)
      if (IsDof(Vids[i]))
        for (size_t j = 0; j < 3; ++j)
          if (IsDof(Vids[j]))
            result[Vids[i]] += v[Vids[j]] * element_mat.coeff(i, j);
  }
  v = result;
}

template <class ForwardOp>
Eigen::SparseMatrix<double> ForwardOperator<ForwardOp>::MatrixSingleScale()
    const {
  if (opts_.cache_forward_mat_) return matrix_;
  return std::move(ComputeMatrixSingleScale());
}

template <class ForwardOp>
Eigen::SparseMatrix<double>
ForwardOperator<ForwardOp>::ComputeMatrixSingleScale() const {
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(triang_.elements().size() * 3);

  auto &vertices = triang_.vertices();
  for (const auto &elem : triang_.elements()) {
    if (!elem->is_leaf()) continue;
    auto &Vids = elem->vertices_view_idx_;
    auto &&element_mat = ForwardOp::ElementMatrix(elem, opts_);

    for (size_t i = 0; i < 3; ++i)
      if (IsDof(Vids[i]))
        for (size_t j = 0; j < 3; ++j)
          if (IsDof(Vids[j]))
            triplets.emplace_back(Vids[i], Vids[j], element_mat(i, j));
  }

  Eigen::SparseMatrix<double> matrix(triang_.V, triang_.V);
  matrix.setFromTriplets(triplets.begin(), triplets.end());
  return matrix;
}

inline Eigen::Matrix3d MassOperator::ElementMatrix(
    const Element2DView *elem, const OperatorOptions &opts) {
  static Eigen::Matrix3d elem_mat =
      (Eigen::Matrix3d() << 2, 1, 1, 1, 2, 1, 1, 1, 2).finished();
  return elem->node()->area() / 12.0 * elem_mat;
}

inline const Eigen::Matrix3d &StiffnessOperator::ElementMatrix(
    const Element2DView *elem, const OperatorOptions &opts) {
  return elem->node()->StiffnessMatrix();
}

inline Eigen::Matrix3d StiffPlusScaledMassOperator::ElementMatrix(
    const Element2DView *elem, const OperatorOptions &opts) {
  // alpha * Stiff + 2^|labda| * Mass.
  return opts.alpha_ * StiffnessOperator::ElementMatrix(elem, opts) +
         (1 << opts.time_level_) * MassOperator::ElementMatrix(elem, opts);
}

BackwardOperator::BackwardOperator(const TriangulationView &triang,
                                   OperatorOptions opts)
    : Operator(triang, opts) {
  std::vector<int> dof_mapping;
  auto &vertices = triang.vertices();
  dof_mapping.reserve(vertices.size());
  for (int i = 0; i < vertices.size(); i++)
    if (IsDof(i)) dof_mapping.push_back(i);
  if (dof_mapping.size() == 0) return;

  std::vector<Eigen::Triplet<double>> triplets, tripletsT;
  triplets.reserve(dof_mapping.size());
  tripletsT.reserve(dof_mapping.size());
  for (int i = 0; i < dof_mapping.size(); i++) {
    triplets.emplace_back(i, dof_mapping[i], 1.0);
    tripletsT.emplace_back(dof_mapping[i], i, 1.0);
  }
  transform_ = Eigen::SparseMatrix<double>(dof_mapping.size(), vertices.size());
  transform_.setFromTriplets(triplets.begin(), triplets.end());
  transformT_ =
      Eigen::SparseMatrix<double>(vertices.size(), dof_mapping.size());
  transformT_.setFromTriplets(tripletsT.begin(), tripletsT.end());
}

void BackwardOperator::ApplyInverseHierarchToSingle(VectorXd &w) const {
  for (size_t vi = triang_.V - 1; vi >= triang_.InitialVertices(); --vi)
    for (auto gp : triang_.Godparents(vi)) w[vi] = w[vi] - 0.5 * w[gp];
}

void BackwardOperator::ApplyTransposeInverseHierarchToSingle(
    VectorXd &w) const {
  for (size_t vi = triang_.InitialVertices(); vi < triang_.V; ++vi)
    for (auto gp : triang_.Godparents(vi))
      if (IsDof(gp)) w[gp] = w[gp] - 0.5 * w[vi];
}

void BackwardOperator::Apply(Eigen::VectorXd &v) const {
  assert(FeasibleVector(v));

  // Hierarchical basis to single scale.
  ApplyTransposeInverseHierarchToSingle(v);
  assert(FeasibleVector(v));

  // Apply in single sale.
  ApplySingleScale(v);
  assert(FeasibleVector(v));

  // Single scale to hierarchical basis.
  ApplyInverseHierarchToSingle(v);
  assert(FeasibleVector(v));
}

// Explicit specializations.
template class ForwardOperator<MassOperator>;
template class ForwardOperator<StiffnessOperator>;
template class ForwardOperator<StiffPlusScaledMassOperator>;

template class DirectInverse<MassOperator>;
template class DirectInverse<StiffnessOperator>;
template class DirectInverse<StiffPlusScaledMassOperator>;

template class MultigridPreconditioner<MassOperator>;
template class MultigridPreconditioner<StiffnessOperator>;
template class MultigridPreconditioner<StiffPlusScaledMassOperator>;

template class XPreconditionerOperator<DirectInverse>;
template class XPreconditionerOperator<MultigridPreconditioner>;

}  // namespace space
