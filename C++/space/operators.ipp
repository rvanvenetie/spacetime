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

  auto &vertices = triang_.vertices();
  for (const auto &elem : triang_.elements()) {
    if (!elem->is_leaf()) continue;
    auto &Vids = elem->vertices_view_idx_;
    auto element_mat = ForwardOp::ElementMatrix(elem, time_level);

    for (size_t i = 0; i < 3; ++i) {
      if (dirichlet_boundary_ && vertices[Vids[i]]->on_domain_boundary)
        continue;
      for (size_t j = 0; j < 3; ++j) {
        if (dirichlet_boundary_ && vertices[Vids[j]]->on_domain_boundary)
          continue;
        triplets.emplace_back(Vids[i], Vids[j], element_mat(i, j));
      }
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
    : BackwardOperator(triang, dirichlet_boundary, time_level),
      triang_mat_(ForwardOp(triang, dirichlet_boundary, time_level)
                      .MatrixSingleScale()),
      // Note that this will leave initial_triang_solver_ with dangling
      // reference, but it doesn't matter for our purpose..
      initial_triang_solver_(triang.InitialTriangulationView(),
                             dirichlet_boundary, time_level) {}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::Prolongate(
    size_t vertex, Eigen::VectorXd &vec_SS) const {
  assert(!triang_.history(vertex).empty());
  for (auto gp : triang_.history(vertex)[0]->RefinementEdge())
    vec_SS[vertex] += 0.5 * vec_SS[gp];
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::Restrict(
    size_t vertex, Eigen::VectorXd &vec_SS) const {
  assert(!triang_.history(vertex).empty());
  for (auto gp : triang_.history(vertex)[0]->RefinementEdge())
    vec_SS[gp] += 0.5 * vec_SS[vertex];
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::RestrictInverse(
    size_t vertex, Eigen::VectorXd &vec_SS) const {
  assert(!triang_.history(vertex).empty());
  for (auto gp : triang_.history(vertex)[0]->RefinementEdge())
    vec_SS[gp] -= 0.5 * vec_SS[vertex];
}

template <typename ForwardOp>
std::vector<std::pair<size_t, double>>
MultigridPreconditioner<ForwardOp>::RowMatrix(
    const MultilevelPatches &mg_triang, size_t vertex) const {
  assert(mg_triang.ContainsVertex(vertex));
  auto &vertices = triang_.vertices();
  if (dirichlet_boundary_) assert(!vertices[vertex]->on_domain_boundary);

  auto &patch = mg_triang.patches()[vertex];
  std::vector<std::pair<size_t, double>> result;
  result.reserve(patch.size() * 3);
  for (auto elem : patch) {
    auto &Vids = elem->vertices_view_idx_;
    auto elem_mat = ForwardOp::ElementMatrix(elem, time_level_);
    for (size_t i = 0; i < 3; ++i) {
      if (Vids[i] != vertex) continue;
      for (size_t j = 0; j < 3; ++j) {
        // If this is the inner product with a boundary dof, skip.
        if (dirichlet_boundary_ && vertices[Vids[j]]->on_domain_boundary)
          continue;
        result.emplace_back(Vids[j], elem_mat(i, j));
      }
    }
  }
  return result;
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::ApplySingleScale(
    Eigen::VectorXd &rhs) const {
  Eigen::SparseMatrix<double> mat_coarse =
      ForwardOp(triang_.InitialTriangulationView(), dirichlet_boundary_)
          .MatrixSingleScale();

  std::cout << mat_coarse << std::endl;
  // TODO: multiple cycles.
  // TODO: V-cycle.
  auto &vertices = triang_.vertices();
  size_t V = vertices.size();
  Eigen::VectorXd rhs_copy = rhs;

  // Take the rhs as initial guess.
  Eigen::VectorXd u = Eigen::VectorXd::Zero(V);

  std::cout << "\nEvaluating MultiGridPreconditioner for rhs given by\n";
  for (int vi = 0; vi < V; vi++)
    std::cout << "\trhs[" << vi << "] = " << rhs[vi] << " for vertex "
              << vertices[vi]->x << ", " << vertices[vi]->y << std::endl;
  std::cout << std::endl;

  // Do a cycle.
  for (size_t cycle = 0; cycle < 10; cycle++) {
    std::cout << " --- Cycle " << cycle + 1 << " --- " << std::endl;

    // alculate the inner products of the current approximation.
    Eigen::VectorXd u_ip = triang_mat_ * u;

    std::cout << "\nCurrent estimated solution  \n";
    for (int vi = 0; vi < V; vi++)
      std::cout << "\tu[" << vi << "] = " << u[vi] << "\tu_ip[" << vi
                << "] = " << u_ip[vi] << std::endl;
    std::cout << std::endl;

    // Create the corrections obtained in this cycle.
    Eigen::VectorXd e = Eigen::VectorXd::Zero(V);

    // Copy back the original rhs.
    rhs = rhs_copy;

    // Step 1: Restrict rhs/u down to the initial mesh.
    for (int vi = V - 1; vi >= triang_.InitialVertices(); --vi) {
      Restrict(vi, rhs);
      Restrict(vi, u_ip);
    }

    // Step 2: Perform an exact solve on this coarsest mesh.
    Eigen::VectorXd e_0 = rhs.head(triang_.InitialVertices()) -
                          u_ip.head(triang_.InitialVertices());
    std::cout << "e_0 before solve" << e_0 << std::endl;
    initial_triang_solver_.ApplySingleScale(e_0);
    std::cout << "e_0 after solve" << e_0 << std::endl;

    // Add this correction.
    e.head(triang_.InitialVertices()) = e_0;

    std::cout << "\nEstimated solution after solving on initial mesh \n";
    for (int vi = 0; vi < V; vi++)
      std::cout << "\tu[" << vi << "]  + e[" << vi << "] = " << u[vi] + e[vi]
                << "\tu_ip[" << vi << "] = " << u_ip[vi] << std::endl;
    std::cout << std::endl;

    // Step 3: Walk back up and do 1-dimensional corrections.
    auto mg_triang = MultilevelPatches::FromCoarsestTriangulation(triang_);
    for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
      assert(mg_triang.CanRefine());
      mg_triang.Refine();

      // Prolongate the current correction to the next level.
      Prolongate(vertex, e);

      // Calculate the RHS/u inner products on the next level.
      RestrictInverse(vertex, rhs);
      RestrictInverse(vertex, u_ip);

      // Find vertex + its grandparents.
      auto grandparents = triang_.history(vertex)[0]->RefinementEdge();
      std::array<size_t, 3> verts{vertex, grandparents[0], grandparents[1]};
      std::cout
          << "\nCalculating corrections for mesh generated by adding vertex "
          << vertex << std::endl;
      std::cout << "\nCurrent estimated solution  \n";
      for (int vi = 0; vi < V; vi++)
        std::cout << "\tu[" << vi << "]  + e[" << vi << "] = " << u[vi] + e[vi]
                  << "\tu_ip[" << vi << "] = " << u_ip[vi] << std::endl;
      std::cout << std::endl;
      // Step 4: Calculate corrections for these three vertices.
      for (size_t vi : verts) {
        // If this vertex is on the boundary, we can simply skip the correction.
        if (dirichlet_boundary_ && vertices[vi]->on_domain_boundary) continue;

        // Get the row of the matrix associated vi on this level.
        auto row_mat = RowMatrix(mg_triang, vi);

        // Calculate a(phi_vi, phi_vi) and a(e, phi_vi).
        double a_phi_vi_phi_vi = 0;
        double a_e_phi_vi = 0;
        for (auto [vj, val] : row_mat) {
          if (dirichlet_boundary_) assert(!vertices[vj]->on_domain_boundary);

          // Calculate the inner product with itself.
          if (vj == vi) a_phi_vi_phi_vi += 1 * val;

          // Calculate the inner product between e and phi_vi.
          a_e_phi_vi += e[vj] * val;
        }

        // Calculate the correction in phi_vi.
        double e_i = (rhs[vi] - u_ip[vi] - a_e_phi_vi) / a_phi_vi_phi_vi;
        std::cout << "Correction at " << vertices[vi]->x << ", "
                  << vertices[vi]->y << "\n\tu_ip[vi] = " << u_ip[vi]
                  << "\n\ta(phi_vi, phi_vi) = " << a_phi_vi_phi_vi
                  << "\n\trhs[vi] = " << rhs[vi]
                  << "\n\ta(e, phi_vi) = " << a_e_phi_vi << std::endl
                  << "\n\te_i = " << e_i << std::endl;

        // Add this correction
        e[vi] += e_i;

        if (dirichlet_boundary_ && vertices[vi]->on_domain_boundary)
          assert(e[vi] == 0 && u[vi] == 0 && u_ip[vi]);
      }
    }

    // Sanity check.
    assert(rhs.isApprox(rhs_copy));

    // Set a new best guess.
    u += e;
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
