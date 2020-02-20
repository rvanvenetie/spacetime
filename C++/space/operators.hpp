#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <boost/range/adaptor/reversed.hpp>

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
  BackwardOperator(const TriangulationView &triang,
                   bool dirichlet_boundary = true);
  virtual Eigen::VectorXd Apply(Eigen::VectorXd vec_in) const final;
  virtual Eigen::VectorXd ApplySinglescale(Eigen::VectorXd vec_SS) const = 0;

 protected:
  // Inverse Hierarhical Basis Transformations.
  Eigen::SparseMatrix<double> transform_;
  Eigen::SparseMatrix<double> transformT_;
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

template <size_t level>
class MassPlusScaledStiffnessOperator : public ForwardOperator {
 public:
  MassPlusScaledStiffnessOperator(const TriangulationView &triang,
                                  bool dirichlet_boundary = true);

 protected:
  using ForwardOperator::matrix_;
  MassOperator mass_;
  StiffnessOperator stiff_;
};

template <typename ForwardOp>
class DirectInverse : public BackwardOperator {
 public:
  DirectInverse(const TriangulationView &triang,
                bool dirichlet_boundary = true);

  Eigen::VectorXd ApplySinglescale(Eigen::VectorXd vec_SS) const final;

 protected:
  ForwardOp forward_op_;
  Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >
      solver_;
};

template <typename ForwardOp>
class CGInverse : public BackwardOperator {
 public:
  CGInverse(const TriangulationView &triang, bool dirichlet_boundary = true);

  Eigen::VectorXd ApplySinglescale(Eigen::VectorXd vec_SS) const final;

 protected:
  ForwardOp forward_op_;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                           Eigen::Lower | Eigen::Upper>
      solver_;
};

}  // namespace space

#include "operators.ipp"
