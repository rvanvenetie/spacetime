#pragma once
#include <boost/range/adaptor/reversed.hpp>
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

  Eigen::VectorXd Solve(const Eigen::VectorXd &x0);
  Eigen::VectorXd Solve() {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(X_delta_.container().size());
    return Solve(x0);
  }

  Eigen::VectorXd Estimate(bool mean_zero = true);

 protected:
  void ApplyMeanZero(
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> *vec);

  TypeXDelta X_delta_, X_delta_underscore_;
  TypeYDelta Y_delta_underscore_;
  HeatEquation heat_eq_;

  TypeGLinForm g_lin_form_;
  TypeU0LinForm u0_lin_form_;
  double theta_;
  size_t saturation_layers_;
};
}  // namespace applications

#include "adaptive_heat_equation.ipp"
