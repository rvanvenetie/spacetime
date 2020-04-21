#pragma once
#include "../datastructures/double_tree_view.hpp"
#include "../spacetime/basis.hpp"
#include "../spacetime/linear_form.hpp"
#include "heat_equation.hpp"

namespace applications {

using datastructures::DoubleNodeVector;
using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using spacetime::GenerateXDeltaUnderscore;
using spacetime::GenerateYDelta;
using spacetime::LinearFormBase;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

struct AdaptiveHeatEquationOptions : public HeatEquationOptions {
  // Solve-step parameters.
  double solve_rtol_ = 1e-5;
  size_t solve_maxit_ = 100;

  // Residual estimation parameter.
  size_t estimate_saturation_layers_ = 1;
  bool estimate_mean_zero_ = true;

  // Dorfler marking parameter.
  double mark_theta_ = 0.7;

  friend std::ostream &operator<<(std::ostream &os,
                                  const AdaptiveHeatEquationOptions &opts) {
    os << "Adaptive heat equation options:" << std::endl;
    os << "\tUse cache: " << (opts.use_cache_ ? "true" : "false") << std::endl;
    os << "\tSolve options -- rtol: " << opts.solve_rtol_
       << "; maxit: " << opts.solve_maxit_ << std::endl;
    os << "\tEstimate options -- saturation layers: "
       << opts.estimate_saturation_layers_
       << "; mean-zero: " << opts.estimate_mean_zero_ << std::endl;
    os << "\tMarking options -- theta: " << opts.mark_theta_ << std::endl;
    os << "\tPreconditioner options:" << std::endl;
    if (opts.P_X_inv_ == HeatEquationOptions::SpaceInverse::DirectInverse)
      os << "\t\tPX: type DirectInverse";
    else if (opts.P_X_inv_ == HeatEquationOptions::SpaceInverse::Multigrid)
      os << "\t\tPX: type Multigrid; cycles " << opts.P_X_mg_cycles_;
    os << "; alpha " << opts.P_X_alpha_ << std::endl;
    if (opts.P_Y_inv_ == HeatEquationOptions::SpaceInverse::DirectInverse)
      os << "\t\tPY: type DirectInverse";
    else if (opts.P_Y_inv_ == HeatEquationOptions::SpaceInverse::Multigrid)
      os << "\t\tPY: type Multigrid; cycles " << opts.P_Y_mg_cycles_;
    os << std::endl;
    return os;
  }
};

class AdaptiveHeatEquation {
 public:
  using TypeXDelta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeYDelta = DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>;
  using TypeXNode = DoubleNodeVector<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeXVector =
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeYVector =
      DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>;
  using TypeXLinForm = LinearFormBase<ThreePointWaveletFn>;
  using TypeYLinForm = LinearFormBase<OrthonormalWaveletFn>;

 public:
  AdaptiveHeatEquation(
      TypeXDelta &&X_delta, std::unique_ptr<TypeYLinForm> &&g_lin_form,
      std::unique_ptr<TypeXLinForm> &&u0_lin_form,
      const AdaptiveHeatEquationOptions &opts = AdaptiveHeatEquationOptions());

  std::pair<TypeXVector *, std::pair<double, int>> Solve(
      const Eigen::VectorXd &x0);
  std::pair<TypeXVector *, std::pair<double, int>> Solve() {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(vec_Xd_in()->container().size());
    return Solve(x0);
  }

  std::pair<TypeXVector *, double> Estimate();

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
  std::unique_ptr<TypeYLinForm> g_lin_form_;
  std::unique_ptr<TypeXLinForm> u0_lin_form_;

  AdaptiveHeatEquationOptions opts_;
};
}  // namespace applications
