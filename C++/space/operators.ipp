#include "operators.hpp"

namespace space {

template <typename ForwardOp>
ForwardMatrix<ForwardOp>::ForwardMatrix(const TriangulationView &triang,
                                        bool dirichlet_boundary,
                                        size_t time_level)
    : ForwardOperator(triang, dirichlet_boundary, time_level),
      matrix_(triang_.V, triang_.V) {
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
    const TriangulationView &triang, size_t cycles, bool dirichlet_boundary,
    size_t time_level)
    : BackwardOperator(triang, dirichlet_boundary, time_level),
      cycles_(cycles),
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
    const MultigridTriangulationView &mg_triang, size_t vertex) const {
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
  // Shortcut.
  size_t V = triang_.V;

  // Take zero vector as initial guess.
  Eigen::VectorXd u = Eigen::VectorXd::Zero(V);
  auto mg_triang = MultigridTriangulationView::FromFinestTriangulation(triang_);

  // Do a V-cycle.
  for (size_t cycle = 0; cycle < cycles_; cycle++) {
    // Copy rhs to avoid rounding error accumulation.
    Eigen::VectorXd rhs_ip = rhs;

    // Part 1: Down-cycle, calculates corrections while coarsening.
    {
      // Calculate the inner products of the previous approx + corrections.
      Eigen::VectorXd u_e_ip = triang_mat_ * u;

      // Keep track of the corrections found in this downward cycle.
      std::vector<double> e;
      e.reserve(V * 3);

      // Step 1: Do a down-cycle and calculate 3 corrections per level.
      for (size_t vertex = V - 1; vertex >= triang_.InitialVertices();
           --vertex) {
        auto grandparents = triang_.history(vertex)[0]->RefinementEdge();
        std::array<size_t, 3> verts{grandparents[1], grandparents[0], vertex};

        // Step 2: Calculate corrections for these three vertices.
        for (size_t vi : verts) {
          // If this vertex is on the boundary, we can simply skip the
          // correction.
          if (dirichlet_boundary_ && triang_.OnBoundary(vi)) {
            e.emplace_back(0.0);
            continue;
          }

          // Get the row of the matrix associated vi on this level.
          auto row_mat = RowMatrix(mg_triang, vi);

          // Calculate a(phi_vi, phi_vi).
          double a_phi_vi_phi_vi = 0;
          for (auto [vj, val] : row_mat)
            if (vj == vi) a_phi_vi_phi_vi += val;

          // Calculate the correction in phi_vi.
          double e_i = (rhs_ip[vi] - u_e_ip[vi]) / a_phi_vi_phi_vi;

          // Add this correction to our estimate.
          e.emplace_back(e_i);

          // Update the inner product with this new correction  <e_k, \cdot>.
          for (auto [vj, val] : row_mat) u_e_ip[vj] += e_i * val;
        }

        // Coarsen mesh, and restrict the inner products calculated thus far.
        mg_triang.Coarsen();
        Restrict(vertex, rhs_ip);
        Restrict(vertex, e_ip);
      }
      assert(!mg_triang.CanCoarsen());

      // Step 3: Do an upward cycle to calculate the correction on finest
      // mesh.
      size_t cnt = 0;
      Eigen::VectorXd e_SS = Eigen::VectorXd::Zero(V);
      for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
        // Prolongate the current correction to the next level.
        Prolongate(vertex, e_SS);

        auto grandparents = triang_.history(vertex)[0]->RefinementEdge();
        std::array<size_t, 3> verts{vertex, grandparents[0], grandparents[1]};
        for (size_t vi : verts) {
          // Add the the correction we have calculated in the above loop,
          // in reversed order of course.
          e_SS[vi] += *(e.rbegin() + cnt);
          cnt++;
        }
      }

      // Step 4: Update u with the new correction.
      u += e_SS;
    }

    // Part 2:  Do exact solve on coarest level and do an up cycle.
    {
      // Step 1: Do a downward-cycle to calculate u_ip on coarsest mesh.
      Eigen::VectorXd u_ip = triang_mat_ * u;
      for (int vertex = V - 1; vertex >= triang_.InitialVertices(); --vertex)
        Restrict(vertex, u_ip);

      // Step 2: Solve on coarsest level.
      assert(!mg_triang.CanCoarsen());
      Eigen::VectorXd e_0 = rhs_ip.head(triang_.InitialVertices()) -
                            u_ip.head(triang_.InitialVertices());
      initial_triang_solver_.ApplySingleScale(e_0);

      // Create vector that will contain corrections in single scale basis.
      Eigen::VectorXd e = Eigen::VectorXd::Zero(V);
      e.head(triang_.InitialVertices()) = e_0;

      // Step 3: Walk back up and do 1-dimensional corrections.
      for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
        assert(mg_triang.CanRefine());
        mg_triang.Refine();

        // Prolongate the current correction to the next level.
        Prolongate(vertex, e);

        // Calculate the RHS/u inner products on the next level.
        RestrictInverse(vertex, rhs_ip);
        RestrictInverse(vertex, u_ip);

        // Find vertex + its grandparents.
        auto grandparents = triang_.history(vertex)[0]->RefinementEdge();
        std::array<size_t, 3> verts{vertex, grandparents[0], grandparents[1]};

        // Step 4: Calculate corrections for these three vertices.
        for (size_t vi : verts) {
          // There is no correction if its not a dof.
          if (dirichlet_boundary_ && triang_.OnBoundary(vi)) continue;

          // Get the row of the matrix associated vi on this level.
          auto row_mat = RowMatrix(mg_triang, vi);

          // Calculate a(phi_vi, phi_vi) and a(e, phi_vi).
          double a_phi_vi_phi_vi = 0;
          double a_e_phi_vi = 0;
          for (auto [vj, val] : row_mat) {
            // Calculate the inner product with itself.
            if (vj == vi) a_phi_vi_phi_vi += 1 * val;

            // Calculate the inner product between e and phi_vi.
            a_e_phi_vi += e[vj] * val;
          }

          // Calculate the correction in phi_vi.
          double e_i = (rhs_ip[vi] - u_ip[vi] - a_e_phi_vi) / a_phi_vi_phi_vi;

          // Add this correction
          e[vi] += e_i;
        }
      }

      // Sanity check.
      assert(rhs_ip.isApprox(rhs));

      // Step 5: Update approximation.
      u += e;
    }
  }

  // Finally: set the approximation as result.
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
