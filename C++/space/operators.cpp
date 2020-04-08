#include "operators.hpp"

#include <vector>

using Eigen::VectorXd;

namespace space {

bool Operator::FeasibleVector(const Eigen::VectorXd &vec) const {
  if (dirichlet_boundary_)
    for (int i = 0; i < triang_.V; ++i)
      if (triang_.OnBoundary(i) && vec[i] != 0) return false;

  return true;
}

void ForwardOperator::Apply(Eigen::VectorXd &v) const {
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

void ForwardOperator::ApplyHierarchToSingle(VectorXd &w) const {
  for (int vi = triang_.InitialVertices(); vi < triang_.V; ++vi)
    for (auto gp : triang_.history(vi)[0]->RefinementEdge())
      w[vi] = w[vi] + 0.5 * w[gp];
}

void ForwardOperator::ApplyTransposeHierarchToSingle(VectorXd &w) const {
  int vi = triang_.V - 1;
  for (; vi >= triang_.InitialVertices(); --vi)
    for (auto gp : triang_.history(vi)[0]->RefinementEdge()) {
      if (dirichlet_boundary_ && triang_.OnBoundary(gp)) continue;
      w[gp] = w[gp] + 0.5 * w[vi];
    }
}

BackwardOperator::BackwardOperator(const TriangulationView &triang,
                                   bool dirichlet_boundary, size_t time_level)
    : Operator(triang, dirichlet_boundary, time_level) {
  std::vector<int> dof_mapping;
  auto &vertices = triang.vertices();
  dof_mapping.reserve(vertices.size());
  for (int i = 0; i < vertices.size(); i++)
    if (!dirichlet_boundary || !triang.OnBoundary(i)) dof_mapping.push_back(i);
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
  int vi = triang_.V - 1;
  for (; vi >= triang_.InitialVertices(); --vi)
    for (auto gp : triang_.history(vi)[0]->RefinementEdge())
      w[vi] = w[vi] - 0.5 * w[gp];
}

void BackwardOperator::ApplyTransposeInverseHierarchToSingle(
    VectorXd &w) const {
  for (int vi = triang_.InitialVertices(); vi < triang_.V; ++vi)
    for (auto gp : triang_.history(vi)[0]->RefinementEdge()) {
      if (dirichlet_boundary_ && triang_.OnBoundary(gp)) continue;
      w[gp] = w[gp] - 0.5 * w[vi];
    }
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

Eigen::Matrix3d MassOperator::ElementMatrix(const Element2DView *elem,
                                            size_t time_level) {
  return elem->node()->area() / 12.0 *
         (Eigen::Matrix3d() << 2, 1, 1, 1, 2, 1, 1, 1, 2).finished();
}

Eigen::Matrix3d StiffnessOperator::ElementMatrix(const Element2DView *elem,
                                                 size_t time_level) {
  Eigen::Vector2d v0, v1, v2;

  v0 << elem->node()->vertices()[0]->x, elem->node()->vertices()[0]->y;
  v1 << elem->node()->vertices()[1]->x, elem->node()->vertices()[1]->y;
  v2 << elem->node()->vertices()[2]->x, elem->node()->vertices()[2]->y;
  Eigen::Matrix<double, 3, 2> D;
  D << v2[0] - v1[0], v2[1] - v1[1], v0[0] - v2[0], v0[1] - v2[1],
      v1[0] - v0[0], v1[1] - v0[1];
  return D * D.transpose() / (4.0 * elem->node()->area());
}

Eigen::Matrix3d StiffPlusScaledMassOperator::ElementMatrix(
    const Element2DView *elem, size_t time_level) {
  return MassOperator::ElementMatrix(elem) +
         pow(2.0, time_level) * StiffnessOperator::ElementMatrix(elem);
}

}  // namespace space
