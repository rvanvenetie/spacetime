#include "adaptive_heat_equation.hpp"

#include <boost/range/adaptor/reversed.hpp>

#include "../tools/linalg.hpp"

namespace applications {
AdaptiveHeatEquation::AdaptiveHeatEquation(
    TypeXDelta &&X_delta, std::unique_ptr<TypeYLinForm> &&g_lin_form,
    std::unique_ptr<TypeXLinForm> &&u0_lin_form,
    const AdaptiveHeatEquationOptions &opts)
    : X_d_(std::move(X_delta)),
      vec_Xd_(
          std::make_shared<TypeXVector>(X_d_.template DeepCopy<TypeXVector>())),
      vec_Xdd_(std::make_shared<TypeXVector>(
          GenerateXDeltaUnderscore(X_d_, opts.estimate_saturation_layers_)
              .template DeepCopy<TypeXVector>())),
      vec_Ydd_(std::make_shared<TypeYVector>(
          GenerateYDelta<DoubleTreeVector, DoubleTreeView>(*vec_Xdd_)
              .template DeepCopy<TypeYVector>())),
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

std::pair<Eigen::VectorXd, double> AdaptiveHeatEquation::Estimate(
    const Eigen::VectorXd &u_dd_d) {
  assert(heat_d_dd_);
  auto A = heat_d_dd_->A();
  auto P_Y = heat_d_dd_->P_Y();
  heat_d_dd_.reset();
  auto heat_dd_dd = HeatEquation(vec_Xdd_, vec_Ydd_, A, P_Y, opts_);

  // Prolongate u_dd_d from X_d to X_dd.
  vec_X_->FromVectorContainer(u_dd_d);
  vec_Xdd_->Reset();
  *vec_Xdd_ += *vec_Xd_;

  auto Su_dd_dd = heat_dd_dd.S()->Apply(vec_Xdd_->ToVectorContainer());

  // Reuse u_dd_dd.
  u_dd_dd->FromVectorContainer(RHS(heat_dd_dd) - Su_dd_dd);
  if (opts_.estimate_mean_zero_) ApplyMeanZero(u_dd_dd);

  auto Xd_nodes = u_dd_dd->Union(*vec_Xd(),
                                 /*call_filter*/ datastructures::func_false);
  assert(Xd_nodes.size() == vec_Xd()->container().size());

  for (auto dblnode : Xd_nodes) dblnode->set_marked(true);
  double sq_norm = 0.0;
  for (auto &dblnode : u_dd_dd->container()) {
    if (dblnode.is_metaroot()) continue;
    if (dblnode.marked()) continue;
    int lvl_diff = std::get<0>(dblnode.nodes())->level() -
                   std::get<1>(dblnode.nodes())->level();
    dblnode.set_value(dblnode.value() / sqrt(1.0 + pow(4.0, lvl_diff)));
    sq_norm += dblnode.value() * dblnode.value();
  }

  for (auto dblnode : Xd_nodes) dblnode->set_marked(false);

  return {u_dd_dd, sqrt(sq_norm)};
}

std::vector<DoubleNodeVector<ThreePointWaveletFn, HierarchicalBasisFn> *>
AdaptiveHeatEquation::Mark() {
  auto nodes = vec_Xdd()->Bfs();
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
    const std::vector<DoubleNodeVector<ThreePointWaveletFn, HierarchicalBasisFn>
                          *> &nodes_to_add) {
  X_d_.ConformingRefinement(*vec_Xdd(), nodes_to_add);

  vec_Xd_ =
      std::make_shared<TypeXVector>(X_d_.template DeepCopy<TypeXVector>());
  TypeXVector vec_Xd_tmp = X_d_.template DeepCopy<TypeXVector>();
  vec_Xd_tmp += *vec_Xd_;
  *vec_Xd_ = std::move(vec_Xd_tmp);

  vec_Xdd_ = std::make_shared<TypeXVector>(
      GenerateXDeltaUnderscore(X_d_, opts_.estimate_saturation_layers_)
          .template DeepCopy<TypeXVector>());
  vec_Xdd_ =
      std::make_shared<TypeXVector>(vec_Xdd_->template DeepCopy<TypeXVector>());

  vec_Ydd_ = std::make_shared<TypeYVector>(
      GenerateYDelta<DoubleTreeVector, DoubleTreeView>(*vec_Xdd_)
          .template DeepCopy<TypeYVector>());
  vec_Ydd_ =
      std::make_shared<TypeYVector>(vec_Ydd_->template DeepCopy<TypeYVector>());

  heat_d_dd_ = std::make_unique<HeatEquation>(vec_Xd_, vec_Xd_, vec_Ydd_,
                                              vec_Ydd_, opts_);
}

void AdaptiveHeatEquation::ApplyMeanZero(
    DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> *vec) {
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
