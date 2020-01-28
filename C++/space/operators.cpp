
#include "operators.hpp"

#include <boost/range/adaptor/reversed.hpp>
#include <vector>

namespace {
using datastructures::TreeVector;
using Eigen::VectorXd;
}  // namespace

namespace space {

VectorXd Operator::ApplyBoundaryConditions(const VectorXd &vec) const {
  VectorXd result{vec};
  const auto &vertices = triang_.vertices();
  for (int i = 0; i < vertices.size(); ++i)
    if (vertices[i]->node()->on_domain_boundary) result[i] = 0;
  return result;
}

void ForwardOperator::Apply(const TreeVector<HierarchicalBasisFn> &vec_in,
                            TreeVector<HierarchicalBasisFn> *vec_out) const {
  VectorXd v{vec_in.ToVector()};

  if (dirichlet_boundary_) v = ApplyBoundaryConditions(v);

  v = ApplyHierarchToSingle(v);
  v = MatrixSingleScale() * v;
  v = ApplyTransposeHierarchToSingle(v);

  if (dirichlet_boundary_) v = ApplyBoundaryConditions(v);

  vec_out->FromVector(v);
}

VectorXd ForwardOperator::ApplyHierarchToSingle(const VectorXd &vec_HB) const {
  VectorXd w{vec_HB};
  for (auto [vi, T] : triang_.history())
    for (auto gp : T->RefinementEdge()) w[vi] = w[vi] + 0.5 * w[gp];
  return w;
}

VectorXd ForwardOperator::ApplyTransposeHierarchToSingle(
    const VectorXd &vec_SS) const {
  VectorXd w{vec_SS};
  for (auto [vi, T] : boost::adaptors::reverse(triang_.history()))
    for (auto gp : T->RefinementEdge()) w[gp] = w[gp] + 0.5 * w[vi];
  return w;
}

MassOperator::MassOperator(const TriangulationView &triang,
                           bool dirichlet_boundary)
    : ForwardOperator(triang, dirichlet_boundary) {
  matrix_ = Eigen::SparseMatrix<double>(triang_.vertices().size(),
                                        triang_.vertices().size());
  Eigen::Matrix3d element_mass;
  element_mass << 2, 1, 1, 1, 2, 1, 1, 1, 2;
  element_mass *= 1.0 / 12.0;

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
