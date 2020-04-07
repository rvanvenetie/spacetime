#include "operators.hpp"

namespace space {

template <typename ForwardOp>
ForwardMatrix<ForwardOp>::ForwardMatrix(const TriangulationView &triang,
                                        bool dirichlet_boundary,
                                        size_t time_level)
    : ForwardOperator(triang, dirichlet_boundary, time_level),
      matrix_(triang_.vertices().size(), triang_.vertices().size()) {
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(triang_.elements().size() * 3);

  for (const auto &elem : triang_.elements()) {
    if (!elem->is_leaf()) continue;
    auto &Vids = elem->vertices_view_idx_;
    auto element_mat = ForwardOp::ElementMatrix(elem, time_level);

    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j) {
        triplets.emplace_back(Vids[i], Vids[j], element_mat(i, j));
      }
  }
  matrix_.setFromTriplets(triplets.begin(), triplets.end());
}

template <typename ForwardOp>
DirectInverse<ForwardOp>::DirectInverse(const TriangulationView &triang,
                                        bool dirichlet_boundary,
                                        size_t time_level)
    : BackwardOperator(triang, dirichlet_boundary, time_level) {
  if (transform_.cols() > 0) {
    auto matrix =
        ForwardOp(triang, dirichlet_boundary, time_level).MatrixSingleScale();
    matrix = transform_ * matrix * transformT_;
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
    : BackwardOperator(triang, dirichlet_boundary, time_level) {
  auto matrix =
      ForwardOp(triang, dirichlet_boundary, time_level).MatrixSingleScale();
  solver_.compute(transform_ * matrix * transformT_);
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
    : BackwardOperator(triang, dirichlet_boundary, time_level) {
  auto coarsest_matrix = ForwardOp(triang.InitialTriangulationView(),
                                   dirichlet_boundary, time_level)
                             .MatrixSingleScale();
  coarsest_solver_.analyzePattern(coarsest_matrix);
  coarsest_solver_.factorize(coarsest_matrix);
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::Prolongate(
    size_t vertex, Eigen::VectorXd &vec_SS) const {
  for (auto gp : triang_.history(vertex)[0]->RefinementEdge())
    vec_SS[vertex] += 0.5 * vec_SS[gp];
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::Restrict(
    size_t vertex, Eigen::VectorXd &vec_SS) const {
  for (auto gp : triang_.history(vertex)[0]->RefinementEdge())
    vec_SS[gp] += 0.5 * vec_SS[vertex];
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::RestrictInverse(
    size_t vertex, Eigen::VectorXd &vec_SS) const {
  for (auto gp : triang_.history(vertex)[0]->RefinementEdge())
    vec_SS[gp] -= 0.5 * vec_SS[vertex];
}

template <typename ForwardOp>
std::vector<std::pair<size_t, double>>
MultigridPreconditioner<ForwardOp>::RowMatrix(
    const MultilevelPatches &mg_triang, size_t vertex) const {
  assert(mg_triang.ContainsVertex(vertex));

  auto &patch = mg_triang.patches()[vertex];
  std::vector<std::pair<size_t, double>> result;
  result.reserve(patch.size() * 3);
  for (auto elem : patch) {
    auto &Vids = elem->vertices_view_idx_;
    auto elem_mat = ForwardOp::ElementMatrix(elem, time_level_);
    for (size_t i = 0; i < 3; ++i) {
      if (Vids[i] != vertex) continue;
      for (size_t j = 0; j < 3; ++j) {
        result.emplace_back(Vids[j], elem_mat(i, j));
      }
    }
  }
  return result;
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::ApplySingleScale(
    Eigen::VectorXd &rhs) const {
  // TODO: multiple cycles.
  // TODO: V-cycle.
  size_t V = triang_.vertices().size();

  // Step 1: Restrict rhs down to the initial mesh.
  int vi = V - 1;
  for (; vi >= triang_.InitialVertices(); --vi) Restrict(vi, rhs);

  // Step 2: Perform an exact solve on this coarsest mesh.
  Eigen::VectorXd rhs_0 = rhs.head(triang_.InitialVertices());
  Eigen::VectorXd u = Eigen::VectorXd::Zero(V);
  u.head(triang_.InitialVertices()) = coarsest_solver_.solve(rhs_0);

  // Step 3: Walk back up.
  auto mg_triang = MultilevelPatches::FromCoarsestTriangulation(triang_);
  for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
    assert(mg_triang.CanRefine());
    mg_triang.Refine();

    // Find vertex + its grandparents.
    auto grandparents = triang_.history(vertex)[0]->RefinementEdge();
    std::array<size_t, 3> verts{vertex, grandparents[0], grandparents[1]};

    // Step 4: Calculate corrections for these three vertices.
    for (size_t vi : verts) {
      // Get the row of the matrix associated vi on this level.
      auto row_mat = RowMatrix(mg_triang, vi);

      // Calculate a(phi_vi, phi_vi) and a(u, phi_vi).
      double a_phi_vi_phi_vi = 0;
      double a_u_phi_vi = 0;
      for (auto [vj, val] : row_mat) {
        // Calculate the inner product with itself.
        if (vj == vi) a_phi_vi_phi_vi += 1 * val;

        // Calculate the inner product between u and phi_vi.
        a_u_phi_vi += u[vj] * val;
      }

      // Calculate the correction in phi_vi.
      double e_i = (rhs[vi] - a_u_phi_vi) / a_phi_vi_phi_vi;
      u[vi] += e_i;
    }

    // Porlongate the current solution to the next level.
    Prolongate(vi, u);

    // Calculate the RHS inner products on the next level.
    RestrictInverse(vi, rhs);
  }

  rhs = u;
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
  stiff_op_.ApplySingleScale(vec_SS);
  inverse_op_.ApplySingleScale(vec_SS);
}

}  // namespace space
