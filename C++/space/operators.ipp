#include "operators.hpp"

namespace space {

template <typename ForwardOp>
DirectInverse<ForwardOp>::DirectInverse(const TriangulationView &triang,
                                        bool dirichlet_boundary,
                                        size_t time_level)
    : BackwardOperator(triang, dirichlet_boundary, time_level),
      forward_op_(triang, dirichlet_boundary, time_level) {
  if (transform_.cols() > 0) {
    auto matrix = transform_ * forward_op_.MatrixSingleScale() * transformT_;
    solver_.analyzePattern(matrix);
    solver_.factorize(matrix);
  }
}

template <typename ForwardOp>
Eigen::VectorXd DirectInverse<ForwardOp>::ApplySinglescale(
    Eigen::VectorXd vec_SS) const {
  if (transform_.cols() > 0) {
    Eigen::VectorXd result = solver_.solve(transform_ * vec_SS);
    return transformT_ * result;
  } else {
    return Eigen::VectorXd::Zero(vec_SS.rows());
  }
}

template <typename ForwardOp>
CGInverse<ForwardOp>::CGInverse(const TriangulationView &triang,
                                bool dirichlet_boundary, size_t time_level)
    : BackwardOperator(triang, dirichlet_boundary, time_level),
      forward_op_(triang, dirichlet_boundary, time_level) {
  solver_.compute(transform_ * forward_op_.MatrixSingleScale() * transformT_);
}

template <typename ForwardOp>
Eigen::VectorXd CGInverse<ForwardOp>::ApplySinglescale(
    Eigen::VectorXd vec_SS) const {
  Eigen::VectorXd result = solver_.solve(transform_ * vec_SS);
  return transformT_ * result;
}

template <template <typename> class InverseOp>
XPreconditionerOperator<InverseOp>::XPreconditionerOperator(
    const TriangulationView &triang, bool dirichlet_boundary, size_t time_level)
    : BackwardOperator(triang, dirichlet_boundary, time_level),
      stiff_op_(triang, dirichlet_boundary, time_level),
      inverse_op_(triang, dirichlet_boundary, time_level) {}

template <template <typename> class InverseOp>
Eigen::VectorXd XPreconditionerOperator<InverseOp>::ApplySinglescale(
    Eigen::VectorXd vec_SS) const {
  return inverse_op_.ApplySinglescale(stiff_op_.MatrixSingleScale() *
                                      inverse_op_.ApplySinglescale(vec_SS));
}

}  // namespace space
