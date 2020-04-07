#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <vector>

#include "basis.hpp"
#include "multilevel_patches.hpp"
#include "triangulation_view.hpp"

namespace space {

class Operator {
 public:
  Operator(const TriangulationView &triang, bool dirichlet_boundary = true,
           size_t time_level = 0)
      : triang_(triang),
        dirichlet_boundary_(dirichlet_boundary),
        time_level_(time_level) {}

  virtual ~Operator() {}

  // Apply the operator in the hierarchical basis.
  virtual void Apply(Eigen::VectorXd &vec_in) const = 0;

 protected:
  const TriangulationView &triang_;
  bool dirichlet_boundary_;
  size_t time_level_;

  // Apply the dirichlet boundary conditions.
  void ApplyBoundaryConditions(Eigen::VectorXd &vec) const;
};

class ForwardOperator : public Operator {
 public:
  using Operator::Operator;

  // Apply the operator in the hierarchical basis.
  virtual void Apply(Eigen::VectorXd &vec_in) const final;

  // Apply the operator in single scale, to be implemented by derived.
  virtual void ApplySingleScale(Eigen::VectorXd &vec_SS) const = 0;

 protected:
  // Hierarhical Basis Transformations from HB to SS, and its transpose.
  void ApplyHierarchToSingle(Eigen::VectorXd &vec_HB) const;
  void ApplyTransposeHierarchToSingle(Eigen::VectorXd &vec_SS) const;
};

// ForwardOperator where apply is done using a sparse matrix.
template <typename ForwardOp>
class ForwardMatrix : public ForwardOperator {
 public:
  ForwardMatrix(const TriangulationView &triang, bool dirichlet_boundary = true,
                size_t time_level = 0);

  virtual void ApplySingleScale(Eigen::VectorXd &vec_SS) const final {
    vec_SS = matrix_ * vec_SS;
  }
  const Eigen::SparseMatrix<double> &MatrixSingleScale() const {
    return matrix_;
  }

 protected:
  Eigen::SparseMatrix<double> matrix_;
};

class BackwardOperator : public Operator {
 public:
  BackwardOperator(const TriangulationView &triang,
                   bool dirichlet_boundary = true, size_t time_level = 0);

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
class MassOperator : public ForwardMatrix<MassOperator> {
 public:
  // Inherit constructor.
  using ForwardMatrix<MassOperator>::ForwardMatrix;

  // Returns the element matrix for the given element.
  static Eigen::Matrix3d ElementMatrix(const Element2DView *elem,
                                       size_t time_level = 0);
};

class StiffnessOperator : public ForwardMatrix<StiffnessOperator> {
 public:
  // Inherit constructor.
  using ForwardMatrix<StiffnessOperator>::ForwardMatrix;

  // Returns the element matrix for the given element.
  static Eigen::Matrix3d ElementMatrix(const Element2DView *elem,
                                       size_t time_level = 0);
};

class StiffPlusScaledMassOperator
    : public ForwardMatrix<StiffPlusScaledMassOperator> {
 public:
  // Inherit constructor.
  using ForwardMatrix<StiffPlusScaledMassOperator>::ForwardMatrix;

  // Returns the element matrix for the given element.
  static Eigen::Matrix3d ElementMatrix(const Element2DView *elem,
                                       size_t time_level);
};

template <typename ForwardOp>
class DirectInverse : public BackwardOperator {
 public:
  DirectInverse(const TriangulationView &triang, bool dirichlet_boundary = true,
                size_t time_level = 0);

  void ApplySingleScale(Eigen::VectorXd &vec_SS) const final;

 protected:
  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
      solver_;
};

template <typename ForwardOp>
class CGInverse : public BackwardOperator {
 public:
  CGInverse(const TriangulationView &triang, bool dirichlet_boundary = true,
            size_t time_level = 0);

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
                          bool dirichlet_boundary = true,
                          size_t time_level = 0);

  void ApplySingleScale(Eigen::VectorXd &vec_SS) const final;

  inline void Prolongate(size_t vertex, Eigen::VectorXd &vec_SS) const;
  inline void Restrict(size_t vertex, Eigen::VectorXd &vec_SS) const;
  inline void RestrictInverse(size_t vertex, Eigen::VectorXd &vec_SS) const;

 protected:
  // Returns a row of the _forward_ matrix on the given multilevel triang.
  // NOTE: The result is not compressed.
  inline std::vector<std::pair<size_t, double>> RowMatrix(
      const MultilevelPatches &mg_triang, size_t vertex) const;

  DirectInverse<ForwardOp> initial_triang_solver_;
};

template <template <typename> class InverseOp>
class XPreconditionerOperator : public BackwardOperator {
 public:
  XPreconditionerOperator(const TriangulationView &triang,
                          bool dirichlet_boundary = true,
                          size_t time_level = 0);

  void ApplySingleScale(Eigen::VectorXd &vec_SS) const final;

 protected:
  StiffnessOperator stiff_op_;
  InverseOp<StiffPlusScaledMassOperator> inverse_op_;
};

}  // namespace space

#include "operators.ipp"
