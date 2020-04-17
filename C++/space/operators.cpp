#include "operators.hpp"

#include <vector>

using Eigen::VectorXd;

namespace space {

bool Operator::FeasibleVector(const Eigen::VectorXd &vec) const {
  for (int i = 0; i < triang_.V; ++i)
    if (!IsDof(i) && vec[i] != 0) return false;

  return true;
}

Eigen::MatrixXd Operator::ToMatrix() const {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(triang_.V, triang_.V);
  for (int i = 0; i < triang_.V; ++i) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(triang_.V);
    if (IsDof(i)) v[i] = 1;
    Apply(v);
    A.col(i) = v;
  }
  return A;
}

void ForwardOperator::Apply(Eigen::VectorXd &v) const {
  assert(FeasibleVector(v));

  // Hierarhical basis to single scale basis.
  ApplyHierarchToSingle(v);
  assert(FeasibleVector(v));

  // Apply the operator in single scale.
  ApplySingleScale(v);
  assert(FeasibleVector(v));

  // Return back to hierarhical basis.
  ApplyTransposeHierarchToSingle(v);
  assert(FeasibleVector(v));
}

void ForwardOperator::ApplyHierarchToSingle(VectorXd &w) const {
  for (size_t vi = triang_.InitialVertices(); vi < triang_.V; ++vi)
    for (auto gp : triang_.Godparents(vi)) w[vi] = w[vi] + 0.5 * w[gp];
}

void ForwardOperator::ApplyTransposeHierarchToSingle(VectorXd &w) const {
  for (size_t vi = triang_.V - 1; vi >= triang_.InitialVertices(); --vi)
    for (auto gp : triang_.Godparents(vi))
      if (IsDof(gp)) w[gp] = w[gp] + 0.5 * w[vi];
}

BackwardOperator::BackwardOperator(const TriangulationView &triang,
                                   OperatorOptions opts)
    : Operator(triang, opts) {
  std::vector<int> dof_mapping;
  auto &vertices = triang.vertices();
  dof_mapping.reserve(vertices.size());
  for (int i = 0; i < vertices.size(); i++)
    if (IsDof(i)) dof_mapping.push_back(i);
  if (dof_mapping.size() == 0) return;

  std::vector<Eigen::Triplet<double>> triplets, tripletsT;
  triplets.reserve(dof_mapping.size());
  tripletsT.reserve(dof_mapping.size());
  for (int i = 0; i < dof_mapping.size(); i++) {
    triplets.emplace_back(i, dof_mapping[i], 1.0);
    tripletsT.emplace_back(dof_mapping[i], i, 1.0);
  }
  transform_ = Eigen::SparseMatrix<double>(dof_mapping.size(), vertices.size());
  transform_.setFromTriplets(triplets.begin(), triplets.end());
  transformT_ =
      Eigen::SparseMatrix<double>(vertices.size(), dof_mapping.size());
  transformT_.setFromTriplets(tripletsT.begin(), tripletsT.end());
}

void BackwardOperator::ApplyInverseHierarchToSingle(VectorXd &w) const {
  for (size_t vi = triang_.V - 1; vi >= triang_.InitialVertices(); --vi)
    for (auto gp : triang_.Godparents(vi)) w[vi] = w[vi] - 0.5 * w[gp];
}

void BackwardOperator::ApplyTransposeInverseHierarchToSingle(
    VectorXd &w) const {
  for (size_t vi = triang_.InitialVertices(); vi < triang_.V; ++vi)
    for (auto gp : triang_.Godparents(vi))
      if (IsDof(gp)) w[gp] = w[gp] - 0.5 * w[vi];
}

void BackwardOperator::Apply(Eigen::VectorXd &v) const {
  assert(FeasibleVector(v));

  // Hierarchical basis to single scale.
  ApplyTransposeInverseHierarchToSingle(v);
  assert(FeasibleVector(v));

  // Apply in single sale.
  ApplySingleScale(v);
  assert(FeasibleVector(v));

  // Single scale to hierarchical basis.
  ApplyInverseHierarchToSingle(v);
  assert(FeasibleVector(v));
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
    auto elem_mat = ForwardOp::ElementMatrix(elem, opts_);
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
        auto godparents = triang_.Godparents(vertex);

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

        auto godparents = triang_.Godparents(vertex);
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
        auto godparents = triang_.Godparents(vertex);

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

// Explicit specializations.
template class DirectInverse<MassOperator>;
template class DirectInverse<StiffnessOperator>;
template class DirectInverse<StiffPlusScaledMassOperator>;

template class MultigridPreconditioner<MassOperator>;
template class MultigridPreconditioner<StiffnessOperator>;
template class MultigridPreconditioner<StiffPlusScaledMassOperator>;

template class XPreconditionerOperator<DirectInverse>;
template class XPreconditionerOperator<MultigridPreconditioner>;

}  // namespace space
