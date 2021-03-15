#define EIGEN_NO_DEBUG
#include "operators.hpp"

#include <boost/range/adaptor/reversed.hpp>
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

template <class ForwardOp>
ForwardOperator<ForwardOp>::ForwardOperator(const TriangulationView &triang,
                                            OperatorOptions opts)
    : Operator(triang, opts) {
  if (opts_.build_mat) InitializeMatrixSingleScale();
}

template <class ForwardOp>
void ForwardOperator<ForwardOp>::Apply(Eigen::VectorXd &v) const {
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

template <class ForwardOp>
void ForwardOperator<ForwardOp>::ApplyHierarchToSingle(VectorXd &w) const {
  for (uint vi = triang_.InitialVertices(); vi < triang_.V; ++vi)
    for (auto gp : triang_.Godparents(vi)) w[vi] = w[vi] + 0.5 * w[gp];
}

template <class ForwardOp>
void ForwardOperator<ForwardOp>::ApplyTransposeHierarchToSingle(
    VectorXd &w) const {
  for (uint vi = triang_.V - 1; vi >= triang_.InitialVertices(); --vi)
    for (auto gp : triang_.Godparents(vi))
      if (IsDof(gp)) w[gp] = w[gp] + 0.5 * w[vi];
}

template <class ForwardOp>
void ForwardOperator<ForwardOp>::ApplySingleScale(Eigen::VectorXd &v) const {
  if (opts_.build_mat) {
    v = matrix_ * v;
  } else {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(v.rows());
    for (const auto &[elem, Vids] : triang_.element_leaves()) {
      auto &&element_mat = ForwardOp::ElementMatrix(elem, opts_);

      for (uint i = 0; i < 3; ++i)
        if (IsDof(Vids[i]))
          for (uint j = 0; j < 3; ++j)
            if (IsDof(Vids[j]))
              result[Vids[i]] += v[Vids[j]] * element_mat.coeff(i, j);
    }
    v = result;
  }
}

template <class ForwardOp>
void ForwardOperator<ForwardOp>::InitializeMatrixSingleScale() {
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(triang_.element_leaves().size() * 3);

  for (const auto &[elem, Vids] : triang_.element_leaves()) {
    auto &&element_mat = ForwardOp::ElementMatrix(elem, opts_);

    for (uint i = 0; i < 3; ++i)
      if (IsDof(Vids[i]))
        for (uint j = 0; j < 3; ++j)
          if (IsDof(Vids[j]))
            triplets.emplace_back(Vids[i], Vids[j], element_mat(i, j));
  }

  matrix_ = Eigen::SparseMatrix<double>(triang_.V, triang_.V);
  matrix_.setFromTriplets(triplets.begin(), triplets.end());
}

inline Eigen::Matrix3d MassOperator::ElementMatrix(
    const Element2D *elem, const OperatorOptions &opts) {
  static Eigen::Matrix3d elem_mat =
      (Eigen::Matrix3d() << 2, 1, 1, 1, 2, 1, 1, 1, 2).finished();
  return elem->area() / 12.0 * elem_mat;
}

inline const Eigen::Matrix3d &StiffnessOperator::ElementMatrix(
    const Element2D *elem, const OperatorOptions &opts) {
  return elem->StiffnessMatrix();
}

inline Eigen::Matrix3d StiffPlusScaledMassOperator::ElementMatrix(
    const Element2D *elem, const OperatorOptions &opts) {
  // alpha * Stiff + 2^|labda| * Mass.
  return opts.alpha * StiffnessOperator::ElementMatrix(elem, opts) +
         (1LL << opts.time_level) * MassOperator::ElementMatrix(elem, opts);
}

BackwardOperator::BackwardOperator(const TriangulationView &triang,
                                   OperatorOptions opts)
    : Operator(triang, opts) {
  std::vector<int> dof_mapping;
  dof_mapping.reserve(triang_.V);
  for (int i = 0; i < triang_.V; i++)
    if (IsDof(i)) dof_mapping.push_back(i);
  if (dof_mapping.size() == 0) return;

  std::vector<Eigen::Triplet<double>> triplets, tripletsT;
  triplets.reserve(dof_mapping.size());
  tripletsT.reserve(dof_mapping.size());
  for (int i = 0; i < dof_mapping.size(); i++) {
    triplets.emplace_back(i, dof_mapping[i], 1.0);
    tripletsT.emplace_back(dof_mapping[i], i, 1.0);
  }
  transform_ = Eigen::SparseMatrix<double>(dof_mapping.size(), triang_.V);
  transform_.setFromTriplets(triplets.begin(), triplets.end());
  transformT_ = Eigen::SparseMatrix<double>(triang_.V, dof_mapping.size());
  transformT_.setFromTriplets(tripletsT.begin(), tripletsT.end());
}

void BackwardOperator::ApplyInverseHierarchToSingle(VectorXd &w) const {
  for (uint vi = triang_.V - 1; vi >= triang_.InitialVertices(); --vi)
    for (auto gp : triang_.Godparents(vi)) w[vi] = w[vi] - 0.5 * w[gp];
}

void BackwardOperator::ApplyTransposeInverseHierarchToSingle(
    VectorXd &w) const {
  for (uint vi = triang_.InitialVertices(); vi < triang_.V; ++vi)
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
    OperatorOptions force_build = opts;
    force_build.build_mat = true;
    auto matrix = ForwardOp(triang, force_build).MatrixSingleScale();
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
  OperatorOptions force_build = opts;
  force_build.build_mat = true;
  auto matrix = ForwardOp(triang, force_build).MatrixSingleScale();
  solver_.compute(transform_ * matrix * transformT_);
}

template <typename ForwardOp>
void CGInverse<ForwardOp>::ApplySingleScale(Eigen::VectorXd &vec_SS) const {
  if (transform_.cols() > 0)
    vec_SS = transformT_ * solver_.solve(transform_ * vec_SS);
  else
    vec_SS.setZero();
}

// Define the class variables.
template <typename ForwardOp>
thread_local std::vector<std::vector<std::pair<uint, double>>>
    MultigridPreconditioner<ForwardOp>::row_mat;
template <typename ForwardOp>
thread_local std::vector<std::vector<Element2D *>>
    MultigridPreconditioner<ForwardOp>::patches;
template <typename ForwardOp>
std::vector<std::vector<uint>>
    MultigridPreconditioner<ForwardOp>::vertices_relaxation;

template <typename ForwardOp>
MultigridPreconditioner<ForwardOp>::MultigridPreconditioner(
    const TriangulationView &triang, OperatorOptions opts)
    : BackwardOperator(triang, opts),
      forward_op_(triang, opts),
      // Note that this will leave initial_triang_solver_ with dangling
      // reference, but it doesn't matter for our purpose..
      initial_triang_solver_(triang.InitialTriangulationView(), opts) {}

namespace {
inline std::array<uint, 3> Vids(Element2D *elem) {
  return {*elem->vertices()[0]->template data<uint>(),
          *elem->vertices()[1]->template data<uint>(),
          *elem->vertices()[2]->template data<uint>()};
}
}  // namespace

template <typename ForwardOp>
inline void MultigridPreconditioner<ForwardOp>::RowMatrix(
    uint vertex, std::vector<std::pair<uint, double>> &result) const {
  assert(IsDof(vertex));

  const auto &patch = patches[vertex];
  result.clear();
  result.reserve(patch.size() * 3);
  for (auto elem : patch) {
    const auto elem_Vids = Vids(elem);
    const auto &elem_mat = ForwardOp::ElementMatrix(elem, opts_);
    for (uint i = 0; i < 3; ++i)
      if (elem_Vids[i] == vertex)
        for (uint j = 0; j < 3; ++j)
          if (IsDof(elem_Vids[j]))
            result.emplace_back(elem_Vids[j], elem_mat.coeff(i, j));
  }
  //  std::sort(
  //      result.begin(), result.end(),
  //      [](const std::pair<uint, double> &p1, const std::pair<uint, double>
  //      &p2) {
  //        return p1.first < p2.first;
  //      });
  //  uint j = 0;
  //  for (uint i = 1; i < result.size(); i++) {
  //    if (result[i].first == result[j].first)
  //      result[j].second += result[i].second;
  //    else
  //      result[++j] = result[i];
  //  }
  //  result.resize(j + 1);
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::InitializeMultigridMatrix() const {
  const uint V = triang_.V;
  patches.resize(V);
  row_mat.resize(V * 3);

  // Store indices in the tree.
  std::vector<uint> indices(V);
  const std::vector<Vertex *> &vertices = triang_.vertices();
  for (uint i = 0; i < V; ++i) {
    indices[i] = i;
    vertices[i]->set_data(&indices[i]);
    patches[i].clear();
    patches[i].reserve(4);
  }

  // Initialize patches var with the triangulation on the finest level.
  for (const auto &[elem, Vids] : triang_.element_leaves())
    for (int vi : Vids) patches[vi].emplace_back(elem);

  // Keep track of vertices included on the current level.
  std::vector<bool> included(V, false);
  vertices_relaxation.resize(triang_.J + 1);
  uint idx = 0;
  for (int k = triang_.J; k >= 1; --k) {
    vertices_relaxation[k].clear();
    for (uint vertex = triang_.VerticesPerLevel(k);
         vertex < triang_.VerticesPerLevel(k + 1); vertex++) {
      const auto godparents = triang_.Godparents(vertex);
      for (uint vi : {vertex, godparents[0], godparents[1]})
        if (!included[vi] && IsDof(vi)) {
          included[vi] = true;
          vertices_relaxation[k].push_back(vi);
          RowMatrix(vi, row_mat[idx++]);
        }
    }

    // Reset the included variable.
    for (uint vertex : vertices_relaxation[k]) included[vertex] = false;

    // Coarsen patches, from k to k - 1.
    for (uint vertex = triang_.VerticesPerLevel(k);
         vertex < triang_.VerticesPerLevel(k + 1); vertex++) {
      for (const auto &elem : vertices[vertex]->parents_element) {
        assert(elem->level() == k - 1);
        for (auto vi : Vids(elem)) {
          // We must remove the children of elem from the patch at vi.
          auto &patch = patches[vi];
          for (const auto &child : elem->children())
            for (int e = 0; e < patch.size(); e++)
              if (patch[e] == child) {
                patch[e] = patch.back();
                patch.resize(patch.size() - 1);
                break;
              }
          // .. and update it with the parent elements.
          patch.emplace_back(elem);
        }
      }
    }
  }

  row_mat.resize(idx);

  // Unset the data stored in the vertices.
  for (auto nv : vertices) nv->reset_data();
}

template <typename ForwardOp>
void MultigridPreconditioner<ForwardOp>::ApplySingleScale(
    Eigen::VectorXd &rhs) const {
  // Shortcut.
  const uint V = triang_.V;

  // Reuse a static variable for storing the row of a matrix.
  static thread_local std::vector<double> e;
  e.reserve(V * 3);

  // Reuse a static variable for storing the residual in the downard cycle.
  static thread_local std::vector<double> r_down;
  r_down.reserve(V * 3);

  // Initialize the multigrid matrix (row_mat).
  InitializeMultigridMatrix();

  // Take zero vector as initial guess.
  Eigen::VectorXd u = Eigen::VectorXd::Zero(V);
  Eigen::VectorXd residual;

  // Do a V-cycle.
  for (size_t cycle = 0; cycle < opts_.mg_cycles; cycle++) {
    // Part 1: Down-cycle, calculates corrections while coarsening.
    {
      // Initialize the residual vector with  a(f, \Phi) - a(u, \Phi).
      residual = u;
      forward_op_.ApplySingleScale(residual);
      residual = rhs - residual;

      // Store all the corrections found in this downward cycle in a vector.
      e.clear();
      r_down.clear();

      // Step 1: Do a down-cycle and calculate 3 corrections per level.
      uint idx = 0;
      for (int k = triang_.J; k >= 1; --k) {
        for (uint vi : vertices_relaxation[k]) {
          // Calculate a(phi_vi, phi_vi).
          double a_phi_vi_phi_vi = 0;
          for (const auto &[vj, val] : row_mat[idx])
            if (vj == vi) a_phi_vi_phi_vi += val;

          // Calculate the correction in phi_vi.
          const double e_vi = residual[vi] / a_phi_vi_phi_vi;

          // Add this correction to our estimate.
          e.emplace_back(e_vi);

          // Store corrected residual.
          r_down.push_back(residual[vi]);

          // Update the residual with this new correction  a(e_vi, \cdot).
          for (const auto &[vj, val] : row_mat[idx]) residual[vj] -= e_vi * val;

          idx++;
        }

        // Restrict the residual calculated thus far.
        for (uint vertex = triang_.VerticesPerLevel(k);
             vertex < triang_.VerticesPerLevel(k + 1); vertex++)
          Restrict(vertex, residual);
      }
    }

    // Part 2:  Do exact solve on coarest level.
    Eigen::VectorXd e_0 = residual.head(triang_.InitialVertices());
    initial_triang_solver_.ApplySingleScale(e_0);

    // Create vector that will contain corrections in single scale basis.
    Eigen::VectorXd e_SS = Eigen::VectorXd::Zero(V);
    e_SS.head(triang_.InitialVertices()) = e_0;

    // Part 3: do an upward cycle
    {
      // Step 1: Walk back up and do 1-dimensional corrections.
      uint idx = row_mat.size() - 1;
      for (int k = 1; k <= triang_.J; k++) {
        // Prolongate the current correction to the next level.
        for (uint vertex = triang_.VerticesPerLevel(k);
             vertex < triang_.VerticesPerLevel(k + 1); vertex++)
          Prolongate(vertex, e_SS);

        // Step 2: Calculate corrections for these vertices.
        for (uint vi : boost::adaptors::reverse(vertices_relaxation[k])) {
          // Add the correction vi found in the downward cycle.
          e_SS[vi] += e[idx];

          // Calculate a(phi_vi, phi_vi) and a(e_SS, phi_vi).
          double a_phi_vi_phi_vi = 0;
          double a_e_phi_vi = 0;
          for (const auto &[vj, val] : row_mat[idx]) {
            // Calculate the inner product with itself.
            if (vj == vi) a_phi_vi_phi_vi += 1 * val;

            // Calculate the inner product between e and phi_vi.
            a_e_phi_vi += e_SS[vj] * val;
          }

          // Calculate the correction in phi_vi.
          const double e_i = (r_down[idx] - a_e_phi_vi) / a_phi_vi_phi_vi;

          // Add this correction
          e_SS[vi] += e_i;

          idx--;
        }
      }
    }

    // Part 3: Update approximation.
    u += e_SS;
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
template class ForwardOperator<MassOperator>;
template class ForwardOperator<StiffnessOperator>;
template class ForwardOperator<StiffPlusScaledMassOperator>;

template class DirectInverse<MassOperator>;
template class DirectInverse<StiffnessOperator>;
template class DirectInverse<StiffPlusScaledMassOperator>;

template class MultigridPreconditioner<MassOperator>;
template class MultigridPreconditioner<StiffnessOperator>;
template class MultigridPreconditioner<StiffPlusScaledMassOperator>;

template class XPreconditionerOperator<DirectInverse>;
template class XPreconditionerOperator<MultigridPreconditioner>;

}  // namespace space
