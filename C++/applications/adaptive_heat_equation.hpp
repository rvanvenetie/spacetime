#pragma once
#include "applications/error_estimator.hpp"
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
  double solve_factor = 3.0;  // factor to divide t_delta by at every cycle.
  double solve_xi = 0.5;
  size_t solve_maxit = 100;

  // Residual estimation parameter.
  size_t estimate_saturation_layers = 1;
  bool estimate_mean_zero = true;

  // Dorfler marking parameter.
  double mark_theta = 0.9;

  friend std::ostream &operator<<(std::ostream &os,
                                  const AdaptiveHeatEquationOptions &opts) {
    os << "Adaptive heat equation options:" << std::endl;
    os << "\tUse cache: " << (opts.use_cache ? "true" : "false") << std::endl;
    os << "\tBuild space matrices: "
       << (opts.build_space_mats ? "true" : "false") << std::endl;
    os << "\tSolve options -- xi: " << opts.solve_xi
       << "; maxit: " << opts.solve_maxit
       << "; division-factor: " << opts.solve_factor << std::endl;
    os << "\tEstimate options -- saturation layers: "
       << opts.estimate_saturation_layers
       << "; mean-zero: " << opts.estimate_mean_zero << std::endl;
    os << "\tMarking options -- theta: " << opts.mark_theta << std::endl;
    os << "\tPreconditioner options:" << std::endl;
    if (opts.PX_inv == HeatEquationOptions::SpaceInverse::DirectInverse)
      os << "\t\tPX: type DirectInverse";
    else if (opts.PX_inv == HeatEquationOptions::SpaceInverse::Multigrid)
      os << "\t\tPX: type Multigrid; cycles " << opts.PX_mg_cycles
         << "; build FW matrix " << (opts.PXY_mg_build ? "true" : "false");
    os << "; alpha " << opts.PX_alpha << std::endl;
    if (opts.PY_inv == HeatEquationOptions::SpaceInverse::DirectInverse)
      os << "\t\tPY: type DirectInverse";
    else if (opts.PY_inv == HeatEquationOptions::SpaceInverse::Multigrid)
      os << "\t\tPY: type Multigrid; cycles " << opts.PY_mg_cycles
         << "; build FW matrix " << (opts.PXY_mg_build ? "true" : "false");
    os << std::endl;
    return os;
  }
};

struct RefineInfo {
  // Data on the nodes that we have marked.
  size_t nodes_marked = 0;
  double res_norm_marked = 0.0;

  // Data on the nodes including the double tree constraint.
  size_t nodes_conforming = 0;
  double res_norm_conforming = 0;
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
      const Eigen::VectorXd &x0, double tol = 1e-5,
      enum tools::linalg::StoppingCriterium crit =
          tools::linalg::StoppingCriterium::Algebraic);
  std::pair<Eigen::VectorXd, tools::linalg::SolverData> Solve(
      double tol = 1e-5, enum tools::linalg::StoppingCriterium crit =
                             tools::linalg::StoppingCriterium::Algebraic) {
    return Solve(Eigen::VectorXd::Zero(vec_Xd_->container().size()), tol, crit);
  }

  std::pair<TypeXVector *, std::pair<double, ErrorEstimator::GlobalError>>
  Estimate(const Eigen::VectorXd &u_dd_d);
  std::vector<TypeXNode *> Mark(TypeXVector *residual);

  // Refines the grid and prolongates a solution living on Xd_.
  RefineInfo Refine(const std::vector<TypeXNode *> &nodes_to_add);

  std::shared_ptr<TypeXVector> vec_Xd() { return vec_Xd_; }
  std::shared_ptr<TypeXVector> vec_Xdd() { return vec_Xdd_; }
  std::shared_ptr<TypeYVector> vec_Ydd() { return vec_Ydd_; }
  HeatEquation *heat_d_dd() { return heat_d_dd_.get(); }

 protected:
  Eigen::VectorXd RHS(HeatEquation &heat);

  std::shared_ptr<TypeXVector> vec_Xd_, vec_Xdd_;
  std::shared_ptr<TypeYVector> vec_Ydd_;
  std::unique_ptr<HeatEquation> heat_d_dd_;
  std::unique_ptr<TypeYLinForm> g_lin_form_;
  std::unique_ptr<TypeXLinForm> u0_lin_form_;

  AdaptiveHeatEquationOptions opts_;
};
}  // namespace applications
