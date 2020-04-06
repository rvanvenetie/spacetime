#include <boost/range/adaptor/reversed.hpp>

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
void DirectInverse<ForwardOp>::ApplySingleScale(Eigen::VectorXd &vec_SS) const {
  if (transform_.cols() > 0)
    vec_SS = transformT_ * solver_.solve(transform_ * vec_SS);
  else
    vec_SS.setZero();
}

template <typename ForwardOp>
CGInverse<ForwardOp>::CGInverse(const TriangulationView &triang,
                                bool dirichlet_boundary, size_t time_level)
    : BackwardOperator(triang, dirichlet_boundary, time_level),
      forward_op_(triang, dirichlet_boundary, time_level) {
  solver_.compute(transform_ * forward_op_.MatrixSingleScale() * transformT_);
}

template <typename ForwardOp>
void CGInverse<ForwardOp>::ApplySingleScale(Eigen::VectorXd &vec_SS) const {
  if (transform_.cols() > 0)
    vec_SS = transformT_ * solver_.solve(transform_ * vec_SS);
  else
    vec_SS.setZero();
}

template <typename ForwardOp>
MultigridPreconditioner<ForwardOp>::MultigridPreconditioner(
    const TriangulationView &triang, bool dirichlet_boundary, size_t time_level,
    size_t cycles)
    : BackwardOperator(triang, dirichlet_boundary, time_level),
      forward_op_(triang, dirichlet_boundary, time_level),
      cycles_(cycles) {
  auto coarsest_matrix = ForwardOp(triang.InitialTriangulationView(),
                                   dirichlet_boundary, time_level)
                             .MatrixSingleScale();
  std::cout << coarsest_matrix << std::endl;
  coarsest_solver_.analyzePattern(coarsest_matrix);
  coarsest_solver_.factorize(coarsest_matrix);
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::ApplySingleScale(
    Eigen::VectorXd &vec_SS) const {
  auto history = triang_.history();
  // Step 1: restrict vec_SS down to the initial mesh.
  for (auto [vi, T] : boost::adaptors::reverse(history))
    for (auto gp : T->RefinementEdge()) vec_SS[gp] += 0.5 * vec_SS[vi];

  // Perform an exact solve on this coarsest mesh.
  Eigen::VectorXd v0 = vec_SS.head(triang_.vertices().size() - history.size());
  Eigen::VectorXd u = coarsest_solver_.solve(v0);
  u.resize(triang_.vertices().size());

  for (size_t i = 0; i < history.size(); i++) {
    auto [vi, T] = history[i];
    // Do stuff on this level.
    double a_u_phi_i = 1.0;      // TODO
    double a_phi_i_phi_i = 1.0;  // TODO
    double c_i = (vec_SS[vi] - a_u_phi_i) / a_phi_i_phi_i;
    u[vi] += c_i;

    for (auto gp : T->RefinementEdge()) {
      // Prolongate RHS and current solution
      vec_SS[gp] -= 0.5 * vec_SS[vi];
      u[gp] -= 0.5 * u[vi];
    }
  }
}

template <template <typename> class InverseOp>
XPreconditionerOperator<InverseOp>::XPreconditionerOperator(
    const TriangulationView &triang, bool dirichlet_boundary, size_t time_level)
    : BackwardOperator(triang, dirichlet_boundary, time_level),
      stiff_op_(triang, dirichlet_boundary, time_level),
      inverse_op_(triang, dirichlet_boundary, time_level) {}

template <template <typename> class InverseOp>
void XPreconditionerOperator<InverseOp>::ApplySingleScale(
    Eigen::VectorXd &vec_SS) const {
  inverse_op_.ApplySingleScale(vec_SS);
  vec_SS = stiff_op_.MatrixSingleScale() * vec_SS;
  inverse_op_.ApplySingleScale(vec_SS);
}

}  // namespace space
