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
    const TriangulationView &triang, bool dirichlet_boundary, size_t time_level)
    : BackwardOperator(triang, dirichlet_boundary, time_level),
      forward_op_(triang, dirichlet_boundary, time_level) {
  auto coarsest_matrix = ForwardOp(triang.InitialTriangulationView(),
                                   dirichlet_boundary, time_level)
                             .MatrixSingleScale();
  coarsest_solver_.analyzePattern(coarsest_matrix);
  coarsest_solver_.factorize(coarsest_matrix);
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::ApplySingleScale(
    Eigen::VectorXd &vec_SS) const {
  // TODO: multiple cycles.
  // TODO: V-cycle.
  size_t V = triang_.vertices().size();

  // Step 1: restrict vec_SS down to the initial mesh.
  int vi = V - 1;
  for (; vi >= triang_.InitialVertices(); --vi)
    for (auto gp : triang_.history(vi)[0]->RefinementEdge())
      vec_SS[gp] += 0.5 * vec_SS[vi];

  // Perform an exact solve on this coarsest mesh.
  Eigen::VectorXd v0 = vec_SS.head(triang_.InitialVertices());
  Eigen::VectorXd u = coarsest_solver_.solve(v0);
  u.resize(V);

  // Now walk back up.
  for (int vi = triang_.InitialVertices(); vi < V; ++vi) {
    // Perform Chinese magic on this level.
    double a_u_phi_i = 1.0;      // TODO
    double a_phi_i_phi_i = 1.0;  // TODO
    double c_i = (vec_SS[vi] - a_u_phi_i) / a_phi_i_phi_i;
    u[vi] += c_i;

    for (auto gp : triang_.history(vi)[0]->RefinementEdge()) {
      // Prolongate RHS and current solution to next level.
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
