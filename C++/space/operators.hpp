#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <vector>

#include "basis.hpp"
#include "multigrid_triangulation_view.hpp"
#include "triangulation_view.hpp"

namespace space {

struct OperatorOptions {
  // Options for all operators.
  bool dirichlet_boundary_ = true;

  // Whether or not to build the matrix for a forward operator.
  bool build_mat_ = true;

  // Options for stiff plus scaled mass operator.
  size_t time_level_ = 0;
  double alpha_ = 1;

  // Options for multigrid preconditioner.
  size_t cycles_ = 5;

  OperatorOptions() = default;
};

class Operator {
 public:
  Operator(const TriangulationView &triang,
           OperatorOptions opts = OperatorOptions())
      : triang_(triang), opts_(std::move(opts)) {}

  virtual ~Operator() {}

  // Apply the operator in the hierarchical basis.
  virtual void Apply(Eigen::VectorXd &vec_in) const = 0;

  // Does the given vertex correspond to a dof?
  inline bool IsDof(size_t vertex) const {
    return !triang_.OnBoundary(vertex) || !opts_.dirichlet_boundary_;
  }

  // Verify that the given vector satisfy the boundary conditions.
  bool FeasibleVector(const Eigen::VectorXd &vec) const;
  inline bool DirichletBoundary() const { return opts_.dirichlet_boundary_; }

  // Overloads required to for Eigen.
  Eigen::VectorXd operator*(const Eigen::VectorXd &vec_in) const {
    Eigen::VectorXd result = vec_in;
    Apply(result);
    return result;
  }
  size_t rows() const { return triang_.V; }
  size_t cols() const { return triang_.V; }

  // Debug function for turning this operator into a mtrix.
  Eigen::MatrixXd ToMatrix() const;

 protected:
  const TriangulationView &triang_;
  OperatorOptions opts_;
};

template <class ForwardOp>
class ForwardOperator : public Operator {
 public:
  ForwardOperator(const TriangulationView &triang,
                  OperatorOptions opts = OperatorOptions());

  // Apply the operator in the hierarchical basis.
  virtual void Apply(Eigen::VectorXd &vec_in) const final;

  // Apply the operator in single scale, to be implemented by derived.
  void ApplySingleScale(Eigen::VectorXd &vec_SS) const;

  const Eigen::SparseMatrix<double> &MatrixSingleScale() const {
    return matrix_;
  }

 protected:
  // Hierarhical Basis Transformations from HB to SS, and its transpose.
  void ApplyHierarchToSingle(Eigen::VectorXd &vec_HB) const;
  void ApplyTransposeHierarchToSingle(Eigen::VectorXd &vec_SS) const;

  void InitializeMatrixSingleScale();
  Eigen::SparseMatrix<double> matrix_;
};

class BackwardOperator : public Operator {
 public:
  BackwardOperator(const TriangulationView &triang,
                   OperatorOptions opts = OperatorOptions());

  // Apply the operator in the hierarchical basis.
  virtual void Apply(Eigen::VectorXd &vec_in) const final;

  // Apply the operator in single scale, to be implemented by derived.
  virtual void ApplySingleScale(Eigen::VectorXd &vec_SS) const = 0;

 protected:
  // Inverse Hierarhical Basis Transformations.
  void ApplyInverseHierarchToSingle(Eigen::VectorXd &vec_SS) const;
  void ApplyTransposeInverseHierarchToSingle(Eigen::VectorXd &vec_HB) const;

  // Vertex-to-DoF transformation matrices.
  Eigen::SparseMatrix<double> transform_;
  Eigen::SparseMatrix<double> transformT_;
};

/**
 *  Implementation of the actual operators.
 */
class MassOperator : public ForwardOperator<MassOperator> {
 public:
  // Inherit constructor.
  using ForwardOperator::ForwardOperator;

  // Returns the element matrix for the given element.
  inline static Eigen::Matrix3d ElementMatrix(const Element2DView *elem,
                                              const OperatorOptions &opts);
};

class StiffnessOperator : public ForwardOperator<StiffnessOperator> {
 public:
  // Inherit constructor.
  using ForwardOperator::ForwardOperator;

  // Returns the element matrix for the given element.
  inline static const Eigen::Matrix3d &ElementMatrix(
      const Element2DView *elem, const OperatorOptions &opts);
};

class StiffPlusScaledMassOperator
    : public ForwardOperator<StiffPlusScaledMassOperator> {
 public:
  // Inherit constructor.
  using ForwardOperator::ForwardOperator;

  // Returns the element matrix for the given element.
  inline static Eigen::Matrix3d ElementMatrix(const Element2DView *elem,
                                              const OperatorOptions &opts);
};

template <typename ForwardOp>
class DirectInverse : public BackwardOperator {
 public:
  DirectInverse(const TriangulationView &triang,
                OperatorOptions opts = OperatorOptions());

  void ApplySingleScale(Eigen::VectorXd &vec_SS) const final;

 protected:
  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
      solver_;
};

template <typename ForwardOp>
class CGInverse : public BackwardOperator {
 public:
  CGInverse(const TriangulationView &triang,
            OperatorOptions opts = OperatorOptions());

  void ApplySingleScale(Eigen::VectorXd &vec_SS) const final;

 protected:
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                           Eigen::Lower | Eigen::Upper>
      solver_;
};

template <typename ForwardOp>
class MultigridPreconditioner : public BackwardOperator {
 public:
  MultigridPreconditioner(const TriangulationView &triang,
                          OperatorOptions opts = OperatorOptions());

  void ApplySingleScale(Eigen::VectorXd &vec_SS) const final;

  inline void Prolongate(size_t vertex, Eigen::VectorXd &vec_SS) const {
    for (auto gp : triang_.Godparents(vertex))
      vec_SS[vertex] += 0.5 * vec_SS[gp];
  }

  inline void Restrict(size_t vertex, Eigen::VectorXd &vec_SS) const {
    for (auto gp : triang_.Godparents(vertex))
      vec_SS[gp] += 0.5 * vec_SS[vertex];
  }

  inline void RestrictInverse(size_t vertex, Eigen::VectorXd &vec_SS) const {
    for (auto gp : triang_.Godparents(vertex))
      vec_SS[gp] -= 0.5 * vec_SS[vertex];
  }

 protected:
  // Returns a row of the _forward_ matrix on the given multilevel triang.
  // NOTE: The result is not compressed.
  void RowMatrix(const MultigridTriangulationView &mg_triang, size_t vertex,
                 std::vector<std::pair<size_t, double>> &result) const;

  // Forward operator on the finest level.
  ForwardOp forward_op_;

  // Solver on the coarsest level.
  DirectInverse<ForwardOp> initial_triang_solver_;
};

template <template <typename> class InverseOp>
class XPreconditionerOperator : public BackwardOperator {
 public:
  XPreconditionerOperator(const TriangulationView &triang,
                          OperatorOptions opts = OperatorOptions());

  void ApplySingleScale(Eigen::VectorXd &vec_SS) const final;

 protected:
  StiffnessOperator stiff_op_;
  InverseOp<StiffPlusScaledMassOperator> inverse_op_;
};

extern template class ForwardOperator<MassOperator>;
extern template class ForwardOperator<StiffnessOperator>;
extern template class ForwardOperator<StiffPlusScaledMassOperator>;

extern template class DirectInverse<MassOperator>;
extern template class DirectInverse<StiffnessOperator>;
extern template class DirectInverse<StiffPlusScaledMassOperator>;

extern template class MultigridPreconditioner<MassOperator>;
extern template class MultigridPreconditioner<StiffnessOperator>;
extern template class MultigridPreconditioner<StiffPlusScaledMassOperator>;

extern template class XPreconditionerOperator<DirectInverse>;
extern template class XPreconditionerOperator<MultigridPreconditioner>;

}  // namespace space
