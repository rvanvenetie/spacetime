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
  using TypeXNode = DoubleNodeVector<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeXVector =
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeYVector =
      DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>;

 public:
  AdaptiveHeatEquation(TypeXDelta &&X_delta, TypeGLinForm &&g_lin_form,
                       TypeU0LinForm &&u0_lin_form, double theta = 0.7,
                       size_t saturation_layers = 1);

  TypeXVector *Solve(const Eigen::VectorXd &x0, double rtol = 1e-5,
                     size_t maxit = 100);
  TypeXVector *Solve(double rtol = 1e-5, size_t maxit = 100) {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(vec_Xd_in()->container().size());
    return Solve(x0, rtol, maxit);
  }

  std::pair<TypeXVector *, double> Estimate(bool mean_zero = true);

  std::vector<TypeXNode *> Mark();
  void Refine(const std::vector<TypeXNode *> &nodes_to_add);

  TypeXVector *vec_Xd_in() { return vec_Xd_in_.get(); }
  TypeXVector *vec_Xd_out() { return vec_Xd_out_.get(); }
  TypeXVector *vec_Xdd_in() { return vec_Xdd_in_.get(); }
  TypeXVector *vec_Xdd_out() { return vec_Xdd_out_.get(); }

 protected:
  Eigen::VectorXd RHS(HeatEquation &heat);
  void ApplyMeanZero(TypeXVector *vec);

  TypeXDelta X_d_;
  std::shared_ptr<TypeXVector> vec_Xd_in_, vec_Xd_out_, vec_Xdd_in_,
      vec_Xdd_out_;
  std::shared_ptr<TypeYVector> vec_Ydd_in_, vec_Ydd_out_;
  std::unique_ptr<HeatEquation> heat_d_dd_;
  TypeGLinForm g_lin_form_;
  TypeU0LinForm u0_lin_form_;

  double theta_;
  size_t saturation_layers_;
};
}  // namespace applications

#include "adaptive_heat_equation.ipp"
