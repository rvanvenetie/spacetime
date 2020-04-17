#include "operators.hpp"

namespace space {

inline Eigen::Matrix3d MassOperator::ElementMatrix(
    const Element2DView *elem, const OperatorOptions &opts) {
  return elem->node()->area() / 12.0 *
         (Eigen::Matrix3d() << 2, 1, 1, 1, 2, 1, 1, 1, 2).finished();
}

inline Eigen::Matrix3d StiffnessOperator::ElementMatrix(
    const Element2DView *elem, const OperatorOptions &opts) {
  Eigen::Vector2d v0, v1, v2;

  v0 << elem->node()->vertices()[0]->x, elem->node()->vertices()[0]->y;
  v1 << elem->node()->vertices()[1]->x, elem->node()->vertices()[1]->y;
  v2 << elem->node()->vertices()[2]->x, elem->node()->vertices()[2]->y;
  Eigen::Matrix<double, 3, 2> D;
  D << v2[0] - v1[0], v2[1] - v1[1], v0[0] - v2[0], v0[1] - v2[1],
      v1[0] - v0[0], v1[1] - v0[1];
  return D * D.transpose() / (4.0 * elem->node()->area());
}

inline Eigen::Matrix3d StiffPlusScaledMassOperator::ElementMatrix(
    const Element2DView *elem, const OperatorOptions &opts) {
  return opts.alpha_ * StiffnessOperator::ElementMatrix(elem, opts) +
         pow(2.0, opts.time_level_) * MassOperator::ElementMatrix(elem, opts);
}

template <typename ForwardOp>
ForwardMatrix<ForwardOp>::ForwardMatrix(const TriangulationView &triang,
                                        OperatorOptions opts)
    : ForwardOperator(triang, opts), matrix_(triang_.V, triang_.V) {
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(triang_.elements().size() * 3);

  auto &vertices = triang_.vertices();
  for (const auto &elem : triang_.elements()) {
    if (!elem->is_leaf()) continue;
    auto &Vids = elem->vertices_view_idx_;
    auto element_mat = ForwardOp::ElementMatrix(elem, opts);

    for (size_t i = 0; i < 3; ++i) {
      if (!IsDof(Vids[i])) continue;
      for (size_t j = 0; j < 3; ++j) {
        if (!IsDof(Vids[j])) continue;
        triplets.emplace_back(Vids[i], Vids[j], element_mat(i, j));
      }
    }
  }
  matrix_.setFromTriplets(triplets.begin(), triplets.end());
}

template <typename ForwardOp>
CGInverse<ForwardOp>::CGInverse(const TriangulationView &triang,
                                OperatorOptions opts)
    : BackwardOperator(triang, opts) {
  auto matrix = ForwardOp(triang, opts).MatrixSingleScale();
  solver_.compute(transform_ * matrix * transformT_);
}

template <typename ForwardOp>
void CGInverse<ForwardOp>::ApplySingleScale(Eigen::VectorXd &vec_SS) const {
  if (transform_.cols() > 0)
    vec_SS = transformT_ * solver_.solve(transform_ * vec_SS);
  else
    vec_SS.setZero();
}

}  // namespace space
