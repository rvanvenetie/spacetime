#pragma once
#include <Eigen/Sparse>

#include "basis.hpp"
#include "triangulation_view.hpp"

namespace space {

class Operator {
 public:
  Operator(const TriangulationView &triang, bool dirichlet_boundary = true)
      : triang_(triang), dirichlet_boundary_(dirichlet_boundary) {}

  virtual ~Operator() {}

  virtual Eigen::VectorXd Apply(Eigen::VectorXd vec_in) const = 0;

 protected:
  const TriangulationView &triang_;
  bool dirichlet_boundary_;

  // Apply the dirichlet boundary conditions.
  void ApplyBoundaryConditions(Eigen::VectorXd &vec) const;
};

class ForwardOperator : public Operator {
 public:
  using Operator::Operator;
  virtual Eigen::VectorXd Apply(Eigen::VectorXd vec_in) const final;

  const Eigen::SparseMatrix<double> &MatrixSingleScale() const {
    return matrix_;
  }

 protected:
  Eigen::SparseMatrix<double> matrix_;

  // Hierarhical Basis Transformations from HB to SS, and its transpose.
  void ApplyHierarchToSingle(Eigen::VectorXd &vec_HB) const;
  void ApplyTransposeHierarchToSingle(Eigen::VectorXd &vec_SS) const;
};

class BackwardOperator : public Operator {
 public:
  using Operator::Operator;
  virtual Eigen::VectorXd Apply(Eigen::VectorXd vec_in) const final;

 protected:
  // Inverse Hierarhical Basis Transformations.
  void ApplyInverseHierarchToSingle(Eigen::VectorXd &vec_SS) const;
  void ApplyTransposeInverseHierarchToSingle(Eigen::VectorXd &vec_HB) const;
};

/**
 *  Implementation of the actual operators.
 */

class MassOperator : public ForwardOperator {
 public:
  MassOperator(const TriangulationView &triang, bool dirichlet_boundary = true);

 protected:
  using ForwardOperator::matrix_;
};

class StiffnessOperator : public ForwardOperator {
 public:
  StiffnessOperator(const TriangulationView &triang,
                    bool dirichlet_boundary = true);

 protected:
  using ForwardOperator::matrix_;
};

}  // namespace space
