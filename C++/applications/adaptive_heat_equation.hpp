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
  using TypeXVector =
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>;

 public:
  AdaptiveHeatEquation(TypeXDelta &&X_delta, TypeGLinForm &&g_lin_form,
                       TypeU0LinForm &&u0_lin_form, double theta = 0.7,
                       size_t saturation_layers = 1);

  TypeXVector *Solve(const Eigen::VectorXd &x0, double rtol = 1e-5,
                     size_t maxit = 100);
  TypeXVector *Solve(double rtol = 1e-5, size_t maxit = 100) {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(X_d_.container().size());
    return Solve(x0, rtol, maxit);
  }

  TypeXVector *Estimate(bool mean_zero = true);

  void Mark();
  void Refine();

  TypeXDelta &X_delta() { return X_d_; }
  TypeXDelta &X_delta_underscore() { return X_dd_; }
  TypeXVector *vec_Xd_in() { return heat_d_dd_.vec_X_in(); }
  TypeXVector *vec_Xd_out() { return heat_d_dd_.vec_X_out(); }
  TypeXVector *vec_Xdd_in() { return heat_dd_dd_.vec_X_in(); }
  TypeXVector *vec_Xdd_out() { return heat_dd_dd_.vec_X_out(); }

 protected:
  Eigen::VectorXd RHS(HeatEquation &heat);
  void ApplyMeanZero(TypeXVector *vec);

  TypeXDelta X_d_, X_dd_;
  TypeYDelta Y_dd_;
  HeatEquation heat_d_dd_;
  HeatEquation heat_dd_dd_;

  TypeGLinForm g_lin_form_;
  TypeU0LinForm u0_lin_form_;
  double theta_;
  size_t saturation_layers_;
};
}  // namespace applications

#include "adaptive_heat_equation.ipp"
