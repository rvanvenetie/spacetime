#pragma once
#include "applications/heat_equation.hpp"
#include "datastructures/double_tree_view.hpp"
#include "spacetime/basis.hpp"
#include "spacetime/linear_form.hpp"
#include "tools/linalg.hpp"

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
      std::shared_ptr<TypeXVector> vec_Xd,
      std::unique_ptr<TypeYLinForm> &&g_lin_form,
      std::unique_ptr<TypeXLinForm> &&u0_lin_form,
      const AdaptiveHeatEquationOptions &opts = AdaptiveHeatEquationOptions());

  std::pair<Eigen::VectorXd, tools::linalg::SolverData> Solve(
      const Eigen::VectorXd &x0);
  std::pair<Eigen::VectorXd, tools::linalg::SolverData> Solve() {
    return Solve(Eigen::VectorXd::Zero(vec_Xd_->container().size()));
  }

  std::pair<TypeXVector *, double> Estimate(const Eigen::VectorXd &u_dd_d);
  std::vector<TypeXNode *> Mark(TypeXVector *residual);

  // Refines the grid and prolongates a solution living on Xd_.
  void Refine(const std::vector<TypeXNode *> &nodes_to_add);

  std::shared_ptr<TypeXVector> vec_Xd() { return vec_Xd_; }
  std::shared_ptr<TypeXVector> vec_Xdd() { return vec_Xdd_; }
  std::shared_ptr<TypeYVector> vec_Ydd() { return vec_Ydd_; }

 protected:
  Eigen::VectorXd RHS(HeatEquation &heat);
  void ApplyMeanZero(TypeXVector *vec);

  std::shared_ptr<TypeXVector> vec_Xd_, vec_Xdd_;
  std::shared_ptr<TypeYVector> vec_Ydd_;
  std::unique_ptr<HeatEquation> heat_d_dd_;
  std::unique_ptr<TypeYLinForm> g_lin_form_;
  std::unique_ptr<TypeXLinForm> u0_lin_form_;

  AdaptiveHeatEquationOptions opts_;
};
}  // namespace applications
