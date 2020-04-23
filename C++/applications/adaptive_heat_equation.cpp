#include "adaptive_heat_equation.hpp"

#include <boost/range/adaptor/reversed.hpp>

#include "../tools/linalg.hpp"

namespace applications {
AdaptiveHeatEquation::AdaptiveHeatEquation(
    const TypeXDelta &X_delta, std::unique_ptr<TypeYLinForm> &&g_lin_form,
    std::unique_ptr<TypeXLinForm> &&u0_lin_form,
    const AdaptiveHeatEquationOptions &opts)
    : vec_Xd_(new TypeXVector(X_delta.template DeepCopy<TypeXVector>())),
      vec_Xdd_(new TypeXVector(GenerateXDeltaUnderscore(
          *vec_Xd_, opts.estimate_saturation_layers_))),
      vec_Ydd_(new TypeYVector(GenerateYDelta<DoubleTreeVector>(*vec_Xdd_))),
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

Eigen::VectorXd AdaptiveHeatEquation::Solve(const Eigen::VectorXd &x0) {
  assert(heat_d_dd_);
  auto [result, data] =
      tools::linalg::PCG(*heat_d_dd_->S(), RHS(*heat_d_dd_), *heat_d_dd_->P_X(),
                         x0, opts_.solve_maxit_, opts_.solve_rtol_);
  return result;
}

auto AdaptiveHeatEquation::Estimate(const Eigen::VectorXd &u_dd_d)
    -> std::pair<TypeXVector *, double> {
  // Create heat equation with X_dd and Y_dd.
  assert(heat_d_dd_);
  auto A = heat_d_dd_->A();
  auto P_Y = heat_d_dd_->P_Y();
  heat_d_dd_.reset();
  auto heat_dd_dd = HeatEquation(vec_Xdd_, vec_Ydd_, A, P_Y, opts_);

  // Prolongate u_dd_d from X_d to X_dd.
  vec_Xd_->FromVectorContainer(u_dd_d);
  vec_Xdd_->FromVector(*vec_Xd_);
  Eigen::VectorXd u_dd_dd = vec_Xdd_->ToVectorContainer();

  // Calculate the residual and store inside the dbltree.
  Eigen::VectorXd residual = RHS(heat_dd_dd) - heat_dd_dd.S()->Apply(u_dd_dd);
  vec_Xdd_->FromVectorContainer(residual);
  if (opts_.estimate_mean_zero_) ApplyMeanZero(vec_Xdd_.get());

  // Get the X_d nodes *inside* X_dd.
  auto vec_Xd_nodes =
      vec_Xdd_->Union(*vec_Xd_,
                      /*call_filter*/ datastructures::func_false);
  assert(vec_Xd_nodes.size() == vec_Xd_->container().size());

  // Do a basis transformation for calculation of the residual wrt Psi_delta.
  for (auto dblnode : vec_Xd_nodes) dblnode->set_marked(true);
  double sq_norm = 0.0;
  for (auto &dblnode : vec_Xdd_->container()) {
    if (dblnode.is_metaroot()) continue;
    if (dblnode.marked()) continue;
    int lvl_diff = std::get<0>(dblnode.nodes())->level() -
                   std::get<1>(dblnode.nodes())->level();
    dblnode.set_value(dblnode.value() / sqrt(1.0 + pow(4.0, lvl_diff)));
    sq_norm += dblnode.value() * dblnode.value();
  }
  for (auto dblnode : vec_Xd_nodes) dblnode->set_marked(false);

  return {vec_Xdd_.get(), sqrt(sq_norm)};
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
  vec_Xd_->ConformingRefinement(*vec_Xdd_, nodes_to_add);

  vec_Xdd_ = std::shared_ptr<TypeXVector>(new TypeXVector(
      GenerateXDeltaUnderscore(*vec_Xd_, opts_.estimate_saturation_layers_)));
  vec_Ydd_ = std::shared_ptr<TypeYVector>(
      new TypeYVector(GenerateYDelta<DoubleTreeVector>(*vec_Xdd_)));

  heat_d_dd_ = std::make_unique<HeatEquation>(vec_Xd_, vec_Ydd_, opts_);
}

void AdaptiveHeatEquation::ApplyMeanZero(TypeXVector *vec) {
  for (auto &dblnode : boost::adaptors::reverse(vec->container())) {
    auto [_, space_node] = dblnode.nodes();
    if (space_node->level() == 0 || space_node->on_domain_boundary()) continue;
    if (std::any_of(space_node->parents().begin(), space_node->parents().end(),
                    [](auto parent) { return parent->on_domain_boundary(); }))
      continue;
    for (auto &parent : dblnode.parents(1)) {
      auto [_, space_parent] = parent->nodes();
      dblnode.set_value(dblnode.value() - 0.5 * space_node->Volume() /
                                              space_parent->Volume() *
                                              parent->value());
    }
  }
}
};  // namespace applications
