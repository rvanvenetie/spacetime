#include "operators.hpp"

#include <boost/range/adaptor/reversed.hpp>
#include <vector>

using Eigen::VectorXd;

namespace space {

void Operator::ApplyBoundaryConditions(VectorXd &vec) const {
  const auto &vertices = triang_.vertices();
  for (int i = 0; i < vertices.size(); ++i)
    if (vertices[i]->on_domain_boundary) vec[i] = 0;
}

Eigen::VectorXd ForwardOperator::Apply(Eigen::VectorXd v) const {
  if (dirichlet_boundary_) ApplyBoundaryConditions(v);

  ApplyHierarchToSingle(v);
  v = MatrixSingleScale() * v;
  ApplyTransposeHierarchToSingle(v);

  if (dirichlet_boundary_) ApplyBoundaryConditions(v);

  return v;
}

void ForwardOperator::ApplyHierarchToSingle(VectorXd &w) const {
  for (auto [vi, T] : triang_.history())
    for (auto gp : T->RefinementEdge()) w[vi] = w[vi] + 0.5 * w[gp];
}

void ForwardOperator::ApplyTransposeHierarchToSingle(VectorXd &w) const {
  for (auto [vi, T] : boost::adaptors::reverse(triang_.history()))
    for (auto gp : T->RefinementEdge()) w[gp] = w[gp] + 0.5 * w[vi];
}

MassOperator::MassOperator(const TriangulationView &triang,
                           bool dirichlet_boundary)
    : ForwardOperator(triang, dirichlet_boundary) {
  matrix_ = Eigen::SparseMatrix<double>(triang_.vertices().size(),
                                        triang_.vertices().size());
  static const Eigen::Matrix3d element_mass =
      1.0 / 12.0 * (Eigen::Matrix3d() << 2, 1, 1, 1, 2, 1, 1, 1, 2).finished();

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(triang_.elements().size() * 3);

  for (const auto &elem : triang_.elements()) {
    if (!elem->is_leaf()) continue;
    auto &Vids = elem->vertices_view_idx_;
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j) {
        triplets.emplace_back(Vids[i], Vids[j],
                              element_mass(i, j) * elem->node()->area());
      }
  }
  matrix_.setFromTriplets(triplets.begin(), triplets.end());
}

}  // namespace space
