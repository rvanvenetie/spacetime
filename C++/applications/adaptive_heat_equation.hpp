#pragma once
#include "../datastructures/double_tree_view.hpp"
#include "../spacetime/basis.hpp"
#include "../tools/linalg.hpp"
#include "heat_equation.hpp"

namespace applications {
using datastructures::DoubleNodeVector;
using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using spacetime::GenerateXDeltaUnderscore;
using spacetime::GenerateYDelta;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;
template <typename TypeGLinForm, typename TypeU0LinForm>
class AdaptiveHeatEquation {
  using TypeXDelta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeYDelta = DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>;

 public:
  AdaptiveHeatEquation(TypeXDelta &&X_delta, TypeGLinForm &&g_lin_form,
                       TypeU0LinForm &&u0_lin_form, double theta = 0.7,
                       size_t saturation_layers = 1);

  Eigen::VectorXd Solve(const Eigen::VectorXd &x0, double rtol = 1e-5,
                        size_t maxit = 100);
  Eigen::VectorXd Solve(double rtol = 1e-5, size_t maxit = 100) {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(X_d_.container().size());
    return Solve(x0, rtol, maxit);
  }

  Eigen::VectorXd Estimate(bool mean_zero = true);

  TypeXDelta &X_delta() { return X_d_; }
  TypeXDelta &X_delta_underscore() { return X_dd_; }
  DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> &vec_Xd_in() {
    return *heat_d_dd_.vec_X_in();
  }
  DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> &vec_Xd_out() {
    return *heat_d_dd_.vec_X_out();
  }

 protected:
  Eigen::VectorXd RHS(HeatEquation &heat);
  void ApplyMeanZero(
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> *vec);

  TypeXDelta X_d_, X_dd_;
  TypeYDelta Y_dd_;
  HeatEquation heat_d_dd_;

  TypeGLinForm g_lin_form_;
  TypeU0LinForm u0_lin_form_;
  double theta_;
  size_t saturation_layers_;
};
}  // namespace applications

#include "adaptive_heat_equation.ipp"
