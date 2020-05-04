#include "adaptive_heat_equation.hpp"

#include <iomanip>

#include "../tools/linalg.hpp"
#include "error_estimator.hpp"

namespace applications {
AdaptiveHeatEquation::AdaptiveHeatEquation(
    std::shared_ptr<TypeXVector> vec_Xd,
    std::unique_ptr<TypeYLinForm> &&g_lin_form,
    std::unique_ptr<TypeXLinForm> &&u0_lin_form,
    const AdaptiveHeatEquationOptions &opts)
    : vec_Xd_(vec_Xd),
      vec_Xdd_(std::make_shared<TypeXVector>(GenerateXDeltaUnderscore(
          *vec_Xd_, opts.estimate_saturation_layers_))),
      vec_Ydd_(std::make_shared<TypeYVector>(
          GenerateYDelta<DoubleTreeVector>(*vec_Xdd_))),
      heat_d_dd_(std::make_unique<HeatEquation>(vec_Xd_, vec_Ydd_, opts)),
      g_lin_form_(std::move(g_lin_form)),
      u0_lin_form_(std::move(u0_lin_form)),
      opts_(opts) {}

Eigen::VectorXd AdaptiveHeatEquation::RHS(HeatEquation &heat) {
  if (opts_.use_cache_)
    heat.B()->Apply(Eigen::VectorXd::Zero(
        heat.B()->cols()));  // This is actually only needed to initialize BT()

  Eigen::VectorXd rhs = g_lin_form_->Apply(heat.vec_Y());
  rhs = heat.P_Y()->Apply(rhs);
  rhs = heat.BT()->Apply(rhs);

  rhs += u0_lin_form_->Apply(heat.vec_X());
  return rhs;
}

std::pair<Eigen::VectorXd, tools::linalg::SolverData>
AdaptiveHeatEquation::Solve(const Eigen::VectorXd &x0) {
  assert(heat_d_dd_);
  return tools::linalg::PCG(*heat_d_dd_->S(), RHS(*heat_d_dd_),
                            *heat_d_dd_->P_X(), x0, opts_.solve_maxit_,
                            opts_.solve_rtol_);
}

auto AdaptiveHeatEquation::Estimate(const Eigen::VectorXd &u_dd_d)
    -> std::pair<TypeXVector *, double> {
  XEquivalentErrorEstimator::ComputeGlobalError(*heat_d_dd_, *g_lin_form_,
                                                *u0_lin_form_, u_dd_d);
  {
    assert(heat_d_dd_);
    auto A = heat_d_dd_->A();
    auto P_Y = heat_d_dd_->P_Y();
    // Invalidate heat_d_dd, we no longer need these bilinear forms.
    heat_d_dd_.reset();

    // Create heat equation with X_dd and Y_dd.
    HeatEquation heat_dd_dd(vec_Xdd_, vec_Ydd_, A, P_Y,
                            /* Ydd_is_GenerateYDelta_Xdd */ true, opts_);

    // Prolongate u_dd_d from X_d to X_dd.
    vec_Xd_->FromVectorContainer(u_dd_d);
    vec_Xdd_->FromVector(*vec_Xd_);
    Eigen::VectorXd u_dd_dd = vec_Xdd_->ToVectorContainer();

    // Calculate the residual and store inside the dbltree.
    Eigen::VectorXd residual = RHS(heat_dd_dd) - heat_dd_dd.S()->Apply(u_dd_dd);
    vec_Xdd_->FromVectorContainer(residual);
    // Let heat_dd_dd go out of scope..
  }

  double global_error = ResidualErrorEstimator::ComputeLocalErrors(
      vec_Xdd_.get(), opts_.estimate_mean_zero_);

  // We know that the residual on Xd should be small, so set it zero explicitly.
  auto vec_Xd_nodes =
      vec_Xdd_->Union(*vec_Xd_,
                      /*call_filter*/ datastructures::func_false);
  assert(vec_Xd_nodes.size() == vec_Xd_->container().size());
  double sum_Xd = 0.0;
  for (auto &dblnode : vec_Xd_nodes) {
    sum_Xd += dblnode->value() * dblnode->value();
    dblnode->set_value(0.0);
  }
  return {vec_Xdd_.get(), sqrt(global_error * global_error - sum_Xd)};
}

auto AdaptiveHeatEquation::Mark(TypeXVector *residual)
    -> std::vector<TypeXNode *> {
  auto nodes = vec_Xdd_->Bfs();
  std::sort(nodes.begin(), nodes.end(), [](auto n1, auto n2) {
    return std::abs(n1->value()) > std::abs(n2->value());
  });
  double sq_norm = 0.0;
  for (auto node : nodes) sq_norm += node->value() * node->value();
  double cur_sq_norm = 0.0;
  size_t last_idx = 0;
  for (; last_idx < nodes.size(); last_idx++) {
    cur_sq_norm += nodes[last_idx]->value() * nodes[last_idx]->value();
    if (cur_sq_norm >= opts_.mark_theta_ * opts_.mark_theta_ * sq_norm) break;
  }
  nodes.resize(last_idx + 1);
  return nodes;
}

void AdaptiveHeatEquation::Refine(
    const std::vector<TypeXNode *> &nodes_to_add) {
  // Refine the solution vector, requires vec_Xdd_.
  vec_Xd_->ConformingRefinement(*vec_Xdd_, nodes_to_add);

  // Reset the objects that we no longer need, this will free the memory.
  heat_d_dd_.reset();
  vec_Xdd_.reset();
  vec_Ydd_.reset();

  vec_Xdd_ = std::make_shared<TypeXVector>(
      GenerateXDeltaUnderscore(*vec_Xd_, opts_.estimate_saturation_layers_));
  vec_Ydd_ = std::make_shared<TypeYVector>(
      GenerateYDelta<DoubleTreeVector>(*vec_Xdd_));

#ifdef VERBOSE
  std::cerr << std::left;
  std::cerr << std::endl << "AdaptiveHeatEquation::Refine" << std::endl;
  std::cerr << "  vec_Xd:  #bfs = " << std::setw(10) << vec_Xd_->Bfs().size()
            << "#container = " << vec_Xd_->container().size() << std::endl;
  std::cerr << "  vec_Xdd: #bfs = " << std::setw(10) << vec_Xdd_->Bfs().size()
            << "#container = " << vec_Xdd_->container().size() << std::endl;
  std::cerr << "  vec_Ydd: #bfs = " << std::setw(10) << vec_Ydd_->Bfs().size()
            << "#container = " << vec_Ydd_->container().size() << std::endl;
  std::cerr << std::right;
#endif

  heat_d_dd_ = std::make_unique<HeatEquation>(vec_Xd_, vec_Ydd_, opts_);
}
};  // namespace applications
