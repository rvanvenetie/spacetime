#include "operators.hpp"

namespace space {

inline Eigen::Matrix3d MassOperator::ElementMatrix(
    const Element2DView *elem, const OperatorOptions &opts) {
  static Eigen::Matrix3d elem_mat =
      (Eigen::Matrix3d() << 2, 1, 1, 1, 2, 1, 1, 1, 2).finished();
  return elem->node()->area() / 12.0 * elem_mat;
}

inline Eigen::Matrix3d StiffnessOperator::ElementMatrix(
    const Element2DView *elem, const OperatorOptions &opts) {
  // The matrix \int_{ref tri} 2 * d/dx phi * d/dx psi.
  static Eigen::Matrix3d elem_mat1 =
      (Eigen::Matrix3d() << 1, -1, 0, -1, 1, 0, 0, 0, 0).finished();
  // The matrix \int_{ref tri} 2* d/dx phi * d/dy psi + 2* d/dy phi * d/dx psi.
  static Eigen::Matrix3d elem_mat2 =
      (Eigen::Matrix3d() << 2, -1, -1, -1, 0, 1, -1, 1, 0).finished();
  // The matrix \int_{ref tri} 2 * d/dy phi * d/dy psi.
  static Eigen::Matrix3d elem_mat3 =
      (Eigen::Matrix3d() << 1, 0, -1, 0, 0, 0, -1, 0, 1).finished();
  auto verts = elem->node()->vertices();
  double v1x = verts[0]->x, v1y = verts[0]->y, v2x = verts[1]->x,
         v2y = verts[1]->y, v3x = verts[2]->x, v3y = verts[2]->y;
  double C1 = (v3x - v1x) * (v3x - v1x) + (v3y - v1y) * (v3y - v1y);
  double C2 = (v2x - v1x) * (v1x - v3x) + (v2y - v1y) * (v1y - v3y);
  double C3 = (v2x - v1x) * (v2x - v1x) + (v2y - v1y) * (v2y - v1y);
  // assert(elem->node()->StiffnessMatrix().isApprox(
  //     1.0 / (4 * elem->node()->area()) *
  //     (elem_mat1 * C1 + elem_mat2 * C2 + elem_mat3 * C3)));
  return 1.0 / (4 * elem->node()->area()) *
         (elem_mat1 * C1 + elem_mat2 * C2 + elem_mat3 * C3);
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
    auto &&element_mat = ForwardOp::ElementMatrix(elem, opts);

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

  const auto &patch = mg_triang.patches()[vertex];
  result.clear();
  result.reserve(patch.size() * 3);
  for (auto elem : patch) {
    const auto &Vids = elem->vertices_view_idx_;
    const auto &elem_mat = ForwardOp::ElementMatrix(elem, opts_);
    for (size_t i = 0; i < 3; ++i)
      if (Vids[i] == vertex)
        for (size_t j = 0; j < 3; ++j)
          if (IsDof(Vids[j]))
            result.emplace_back(Vids[j], elem_mat.coeff(i, j));
  }
  std::sort(
      result.begin(), result.end(),
      [](const std::pair<size_t, double> &p1,
         const std::pair<size_t, double> &p2) { return p1.first < p2.first; });
  size_t j = 0;
  for (size_t i = 1; i < result.size(); i++) {
    if (result[i].first == result[j].first)
      result[j].second += result[i].second;
    else
      result[++j] = result[i];
  }
  result.resize(j + 1);
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::ApplySingleScale(
    Eigen::VectorXd &rhs) const {
  // Shortcut.
  const size_t V = triang_.V;

  // Reuse a static variable for storing the row of a matrix.
  static std::vector<std::vector<std::pair<size_t, double>>> row_mat;
  static std::vector<double> e;
  e.reserve(V * 3);

  // First, initialize the row matrix variables.
  {
    auto mg_triang =
        MultigridTriangulationView::FromFinestTriangulation(triang_);
    row_mat.resize(V * 3);
    size_t idx = 0;
    for (size_t vertex = V - 1; vertex >= triang_.InitialVertices(); --vertex) {
      const auto godparents = triang_.Godparents(vertex);
      for (size_t vi : {godparents[1], godparents[0], vertex}) {
        if (IsDof(vi)) RowMatrix(mg_triang, vi, row_mat[idx++]);
      }
      mg_triang.Coarsen();
    }
    row_mat.resize(idx);
  }

  // Take zero vector as initial guess.
  Eigen::VectorXd u = Eigen::VectorXd::Zero(V);

  // Do a V-cycle.
  for (size_t cycle = 0; cycle < opts_.cycles_; cycle++) {
    // Part 1: Down-cycle, calculates corrections while coarsening.
    {
      // Initialize the residual vector with  a(f, \Phi) - a(u, \Phi).
      Eigen::VectorXd residual = rhs - triang_mat_ * u;

      // Store all the corrections found in this downward cycle in a vector.
      e.clear();

      // Step 1: Do a down-cycle and calculate 3 corrections per level.
      size_t idx = 0;
      for (size_t vertex = V - 1; vertex >= triang_.InitialVertices();
           --vertex) {
        const auto godparents = triang_.Godparents(vertex);

        // Step 2: Calculate corrections for these three vertices.
        for (size_t vi : {godparents[1], godparents[0], vertex}) {
          // If this vertex is on the boundary, we can simply skip the
          // correction.
          if (!IsDof(vi)) continue;

          // Calculate a(phi_vi, phi_vi).
          double a_phi_vi_phi_vi = 0;
          for (auto [vj, val] : row_mat[idx])
            if (vj == vi) a_phi_vi_phi_vi += val;

          // Calculate the correction in phi_vi.
          const double e_vi = residual[vi] / a_phi_vi_phi_vi;

          // Add this correction to our estimate.
          e.emplace_back(e_vi);

          // Update the residual with this new correction  a(e_vi, \cdot).
          for (auto [vj, val] : row_mat[idx++]) residual[vj] -= e_vi * val;
        }

        // Coarsen mesh, and restrict the residual calculated thus far.
        Restrict(vertex, residual);
      }

      // Step 3: Do an upward cycle to calculate the correction on finest mesh.
      Eigen::VectorXd e_SS = Eigen::VectorXd::Zero(V);
      idx = e.size();
      assert(e.size() == row_mat.size());
      for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
        // Prolongate the current correction to the next level.
        Prolongate(vertex, e_SS);

        const auto godparents = triang_.Godparents(vertex);
        for (size_t vi : {vertex, godparents[0], godparents[1]}) {
          if (!IsDof(vi)) continue;

          // Add the the correction we have calculated in the above loop, in
          // reversed order of course.
          e_SS[vi] += e[--idx];
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
      size_t idx = row_mat.size();
      for (size_t vertex = triang_.InitialVertices(); vertex < V; ++vertex) {
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

          // Calculate a(phi_vi, phi_vi) and a(e_SS, phi_vi).
          double a_phi_vi_phi_vi = 0;
          double a_e_phi_vi = 0;
          for (auto [vj, val] : row_mat[--idx]) {
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
