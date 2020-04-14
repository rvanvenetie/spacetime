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
    const TriangulationView &triang, bool dirichlet_boundary, size_t time_level,
    size_t cycles)
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
  assert(IsDof(vertex));

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
        if (IsDof(Vids[j])) result.emplace_back(Vids[j], elem_mat(i, j));
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
    // Part 1: Down-cycle, calculates corrections while coarsening.
    {
      // Initialize the residual vector with  a(f, \Phi) - a(u, \Phi).
      Eigen::VectorXd residual = rhs - triang_mat_ * u;

      // Store all the corrections found in this downward cycle in a vector.
      std::vector<double> e;
      e.reserve(V * 3);

      // Step 1: Do a down-cycle and calculate 3 corrections per level.
      for (size_t vertex = V - 1; vertex >= triang_.InitialVertices();
           --vertex) {
        auto godparents = triang_.history(vertex)[0]->RefinementEdge();

        // Step 2: Calculate corrections for these three vertices.
        for (size_t vi : {godparents[1], godparents[0], vertex}) {
          // If this vertex is on the boundary, we can simply skip the
          // correction.
          if (!IsDof(vi)) {
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
          double e_vi = residual[vi] / a_phi_vi_phi_vi;

          // Add this correction to our estimate.
          e.emplace_back(e_vi);

          // Update the residual with this new correction  a(e_vi, \cdot).
          for (auto [vj, val] : row_mat) residual[vj] -= e_vi * val;
        }

        // Coarsen mesh, and restrict the residual calculated thus far.
        mg_triang.Coarsen();
        Restrict(vertex, residual);
      }
      assert(!mg_triang.CanCoarsen());

      // Step 3: Do an upward cycle to calculate the correction on finest mesh.
      int rev_vi = e.size() - 1;
      Eigen::VectorXd e_SS = Eigen::VectorXd::Zero(V);
      for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
        // Prolongate the current correction to the next level.
        Prolongate(vertex, e_SS);

        auto godparents = triang_.history(vertex)[0]->RefinementEdge();
        for (size_t vi : {vertex, godparents[0], godparents[1]}) {
          // Add the the correction we have calculated in the above loop, in
          // reversed order of course.
          e_SS[vi] += e[rev_vi];
          rev_vi--;
        }
      }

      // Step 4: Update u with the new correction.
      u += e_SS;
    }

    // Part 2:  Do exact solve on coarest level and do an up cycle.
    {
      // Initialize the residual vector with  a(f, \Phi) - a(u, \Phi).
      Eigen::VectorXd residual = rhs - triang_mat_ * u;

      // Step 1: Do a downward-cycle restrict the residual on the coarsest mesh.
      for (int vertex = V - 1; vertex >= triang_.InitialVertices(); --vertex)
        Restrict(vertex, residual);

      // Step 2: Solve on coarsest level.
      assert(!mg_triang.CanCoarsen());
      Eigen::VectorXd e_0 = residual.head(triang_.InitialVertices());
      initial_triang_solver_.ApplySingleScale(e_0);

      // Create vector that will contain corrections in single scale basis.
      Eigen::VectorXd e_SS = Eigen::VectorXd::Zero(V);
      e_SS.head(triang_.InitialVertices()) = e_0;

      // Step 3: Walk back up and do 1-dimensional corrections.
      for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
        assert(mg_triang.CanRefine());
        mg_triang.Refine();

        // Prolongate the current correction to the next level.
        Prolongate(vertex, e_SS);

        // Calculate the residual on the next level.
        RestrictInverse(vertex, residual);

        // Find vertex + its godparents.
        auto godparents = triang_.history(vertex)[0]->RefinementEdge();

        // Step 4: Calculate corrections for these three vertices.
        for (size_t vi : {vertex, godparents[0], godparents[1]}) {
          // There is no correction if its not a dof.
          if (!IsDof(vi)) continue;

          // Get the row of the matrix associated vi on this level.
          auto row_mat = RowMatrix(mg_triang, vi);

          // Calculate a(phi_vi, phi_vi) and a(e_SS, phi_vi).
          double a_phi_vi_phi_vi = 0;
          double a_e_phi_vi = 0;
          for (auto [vj, val] : row_mat) {
            // Calculate the inner product with itself.
            if (vj == vi) a_phi_vi_phi_vi += 1 * val;

            // Calculate the inner product between e and phi_vi.
            a_e_phi_vi += e_SS[vj] * val;
          }

          // Calculate the correction in phi_vi.
          double e_i = (residual[vi] - a_e_phi_vi) / a_phi_vi_phi_vi;

          // Add this correction
          e_SS[vi] += e_i;
        }
      }

      // Step 5: Update approximation.
      u += e_SS;
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
