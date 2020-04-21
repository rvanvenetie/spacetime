#include "operators.hpp"

#ifndef EIGEN_NO_DEBUG
#define COMPILE_WITH_EIGEN_DEBUG
#define EIGEN_NO_DEBUG
#endif

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
DirectInverse<ForwardOp>::DirectInverse(const TriangulationView &triang,
                                        OperatorOptions opts)
    : BackwardOperator(triang, opts) {
  if (transform_.cols() > 0) {
    auto matrix = ForwardOp(triang, opts).MatrixSingleScale();
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

template <typename ForwardOp>
MultigridPreconditioner<ForwardOp>::MultigridPreconditioner(
    const TriangulationView &triang, OperatorOptions opts)
    : BackwardOperator(triang, opts),
      triang_mat_(ForwardOp(triang, opts).MatrixSingleScale()),
      // Note that this will leave initial_triang_solver_ with dangling
      // reference, but it doesn't matter for our purpose..
      initial_triang_solver_(triang.InitialTriangulationView(), opts) {}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::Prolongate(
    size_t vertex, Eigen::VectorXd &vec_SS) const {
  for (auto gp : triang_.Godparents(vertex)) vec_SS[vertex] += 0.5 * vec_SS[gp];
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::Restrict(
    size_t vertex, Eigen::VectorXd &vec_SS) const {
  for (auto gp : triang_.Godparents(vertex)) vec_SS[gp] += 0.5 * vec_SS[vertex];
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::RestrictInverse(
    size_t vertex, Eigen::VectorXd &vec_SS) const {
  for (auto gp : triang_.Godparents(vertex)) vec_SS[gp] -= 0.5 * vec_SS[vertex];
}

template <typename ForwardOp>
inline void MultigridPreconditioner<ForwardOp>::RowMatrix(
    const MultigridTriangulationView &mg_triang, size_t vertex,
    std::vector<std::pair<size_t, double>> &result) const {
  assert(mg_triang.ContainsVertex(vertex));
  assert(IsDof(vertex));

  auto &patch = mg_triang.patches()[vertex];
  result.clear();
  result.reserve(patch.size() * 3);
  for (auto elem : patch) {
    auto &Vids = elem->vertices_view_idx_;
    auto elem_mat = ForwardOp::ElementMatrix(elem, opts_);
    for (size_t i = 0; i < 3; ++i) {
      if (Vids[i] != vertex) continue;
      for (size_t j = 0; j < 3; ++j) {
        // If this is the inner product with a boundary dof, skip.
        if (IsDof(Vids[j])) result.emplace_back(Vids[j], elem_mat(i, j));
      }
    }
  }
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::ApplySingleScale(
    Eigen::VectorXd &rhs) const {
  // Reuse a static variable for storing the row of a matrix.
  static std::vector<std::pair<size_t, double>> row_mat;

  // Shortcut.
  const size_t V = triang_.V;

  // Take zero vector as initial guess.
  Eigen::VectorXd u = Eigen::VectorXd::Zero(V);
  auto mg_triang = MultigridTriangulationView::FromFinestTriangulation(triang_);

  // Do a V-cycle.
  for (size_t cycle = 0; cycle < opts_.cycles_; cycle++) {
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
        const auto godparents = triang_.Godparents(vertex);

        // Step 2: Calculate corrections for these three vertices.
        for (size_t vi : {godparents[1], godparents[0], vertex}) {
          // If this vertex is on the boundary, we can simply skip the
          // correction.
          if (!IsDof(vi)) {
            e.emplace_back(0.0);
            continue;
          }

          // Get the row of the matrix associated vi on this level.
          RowMatrix(mg_triang, vi, row_mat);

          // Calculate a(phi_vi, phi_vi).
          double a_phi_vi_phi_vi = 0;
          for (auto [vj, val] : row_mat)
            if (vj == vi) a_phi_vi_phi_vi += val;

          // Calculate the correction in phi_vi.
          const double e_vi = residual[vi] / a_phi_vi_phi_vi;

          // Add this correction to our estimate.
          e.emplace_back(e_vi);

          // Update the residual with this new correction  a(e_vi, \cdot).
          for (auto [vj, val] : row_mat) residual[vj] -= e_vi * val;
        }

        // Coarsen mesh, and restrict the residual calculated thus far.
        mg_triang.Coarsen();
        Restrict(vertex, residual);
      }

      // Step 3: Do an upward cycle to calculate the correction on finest mesh.
      int rev_vi = e.size() - 1;
      Eigen::VectorXd e_SS = Eigen::VectorXd::Zero(V);
      for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
        // Prolongate the current correction to the next level.
        Prolongate(vertex, e_SS);

        const auto godparents = triang_.Godparents(vertex);
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
      for (size_t vertex = V - 1; vertex >= triang_.InitialVertices(); --vertex)
        Restrict(vertex, residual);

      // Step 2: Solve on coarsest level.
      Eigen::VectorXd e_0 = residual.head(triang_.InitialVertices());
      initial_triang_solver_.ApplySingleScale(e_0);

      // Create vector that will contain corrections in single scale basis.
      Eigen::VectorXd e_SS = Eigen::VectorXd::Zero(V);
      e_SS.head(triang_.InitialVertices()) = e_0;

      // Step 3: Walk back up and do 1-dimensional corrections.
      for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
        mg_triang.Refine();

        // Prolongate the current correction to the next level.
        Prolongate(vertex, e_SS);

        // Calculate the residual on the next level.
        RestrictInverse(vertex, residual);

        // Find vertex + its godparents.
        const auto godparents = triang_.Godparents(vertex);

        // Step 4: Calculate corrections for these three vertices.
        for (size_t vi : {vertex, godparents[0], godparents[1]}) {
          // There is no correction if its not a dof.
          if (!IsDof(vi)) continue;

          // Get the row of the matrix associated vi on this level.
          RowMatrix(mg_triang, vi, row_mat);

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
          const double e_i = (residual[vi] - a_e_phi_vi) / a_phi_vi_phi_vi;

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
    const TriangulationView &triang, OperatorOptions opts)
    : BackwardOperator(triang, opts),
      stiff_op_(triang, opts),
      inverse_op_(triang, opts) {}

template <template <typename> class InverseOp>
void XPreconditionerOperator<InverseOp>::ApplySingleScale(
    Eigen::VectorXd &vec_SS) const {
  inverse_op_.ApplySingleScale(vec_SS);
  stiff_op_.ApplySingleScale(vec_SS);
  inverse_op_.ApplySingleScale(vec_SS);
}

}  // namespace space

#ifdef COMPILE_WITH_EIGEN_DEBUG
#undef EIGEN_NO_DEBUG
#undef COMPILE_WITH_EIGEN_DEBUG
#endif
