#include <boost/range/adaptor/reversed.hpp>
#include "operators.hpp"

namespace space {

template <size_t level>
MassPlusScaledStiffnessOperator<level>::MassPlusScaledStiffnessOperator(
    const TriangulationView &triang, bool dirichlet_boundary)
    : ForwardOperator(triang, dirichlet_boundary),
      mass_(triang, dirichlet_boundary),
      stiff_(triang, dirichlet_boundary) {
  matrix_ =
      stiff_.MatrixSingleScale() + pow(2, level) * mass_.MatrixSingleScale();
}

template <typename ForwardOp>
DirectInverse<ForwardOp>::DirectInverse(const TriangulationView &triang,
                                        bool dirichlet_boundary)
    : BackwardOperator(triang, dirichlet_boundary),
      forward_op_(triang, dirichlet_boundary) {
  auto matrix = transform_ * forward_op_.MatrixSingleScale() * transformT_;
  solver_.analyzePattern(matrix);
  solver_.factorize(matrix);
}

template <typename ForwardOp>
Eigen::VectorXd DirectInverse<ForwardOp>::ApplySinglescale(
    Eigen::VectorXd vec_SS) const {
  Eigen::VectorXd result = solver_.solve(transform_ * vec_SS);
  return transformT_ * result;
}

template <typename ForwardOp>
CGInverse<ForwardOp>::CGInverse(const TriangulationView &triang,
                                bool dirichlet_boundary)
    : BackwardOperator(triang, dirichlet_boundary),
      forward_op_(triang, dirichlet_boundary) {
  solver_.compute(transform_ * forward_op_.MatrixSingleScale() * transformT_);
}

template <typename ForwardOp>
Eigen::VectorXd CGInverse<ForwardOp>::ApplySinglescale(
    Eigen::VectorXd vec_SS) const {
  Eigen::VectorXd result = solver_.solve(transform_ * vec_SS);
  return transformT_ * result;
}

}  // namespace space
