#include "adaptive_heat_equation.hpp"

#include <boost/range/adaptor/reversed.hpp>

#include "../tools/linalg.hpp"

namespace applications {
AdaptiveHeatEquation::AdaptiveHeatEquation(
    TypeXDelta &&X_delta, std::unique_ptr<TypeYLinForm> &&g_lin_form,
    std::unique_ptr<TypeXLinForm> &&u0_lin_form, double theta,
    size_t saturation_layers)
    : X_d_(std::move(X_delta)),
      vec_Xd_in_(
          std::make_shared<TypeXVector>(X_d_.template DeepCopy<TypeXVector>())),
      vec_Xd_out_(
          std::make_shared<TypeXVector>(X_d_.template DeepCopy<TypeXVector>())),
      vec_Xdd_in_(std::make_shared<TypeXVector>(
          GenerateXDeltaUnderscore(X_d_, saturation_layers)
              .template DeepCopy<TypeXVector>())),
      vec_Xdd_out_(std::make_shared<TypeXVector>(
          vec_Xdd_in_->template DeepCopy<TypeXVector>())),
      vec_Ydd_in_(std::make_shared<TypeYVector>(
          GenerateYDelta(*vec_Xdd_in_).template DeepCopy<TypeYVector>())),
      vec_Ydd_out_(std::make_shared<TypeYVector>(
          vec_Ydd_in_->template DeepCopy<TypeYVector>())),
      heat_d_dd_(std::make_unique<HeatEquation>(vec_Xd_in_, vec_Xd_out_,
                                                vec_Ydd_in_, vec_Ydd_out_)),
      g_lin_form_(std::move(g_lin_form)),
      u0_lin_form_(std::move(u0_lin_form)),
      theta_(theta),
      saturation_layers_(saturation_layers) {}

Eigen::VectorXd AdaptiveHeatEquation::RHS(HeatEquation &heat) {
  heat.B()->Apply();  // This is actually only needed to initialize BT()
  g_lin_form_->Apply(heat.vec_Y_out());
  heat.P_Y()->Apply();
  auto rhs = heat.BT()->Apply();
  rhs += u0_lin_form_->Apply(heat.vec_X_in());
  return rhs;
}

DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>
    *AdaptiveHeatEquation::Solve(const Eigen::VectorXd &x0, double rtol,
                                 size_t maxit) {
  assert(heat_d_dd_);
  auto [result, data] = tools::linalg::PCG(*heat_d_dd_->S(), RHS(*heat_d_dd_),
                                           *heat_d_dd_->P_X(), x0, maxit, rtol);
  vec_Xd_out()->FromVectorContainer(result);
  return vec_Xd_out();
}

std::pair<DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> *, double>
AdaptiveHeatEquation::Estimate(bool mean_zero) {
  assert(heat_d_dd_);
  auto A = heat_d_dd_->A();
  auto P_Y = heat_d_dd_->P_Y();
  heat_d_dd_.reset();
  auto heat_dd_dd = HeatEquation(vec_Xdd_in_, vec_Xdd_out_, vec_Ydd_in_,
                                 vec_Ydd_out_, A, P_Y);
  auto u_dd_dd = vec_Xdd_in();
  u_dd_dd->Reset();
  *u_dd_dd += *vec_Xd_out();
  auto Su_dd_dd = heat_dd_dd.S()->Apply();

  // Reuse u_dd_dd.
  u_dd_dd->FromVectorContainer(RHS(heat_dd_dd) - Su_dd_dd);
  if (mean_zero) ApplyMeanZero(u_dd_dd);

  auto Xd_nodes = u_dd_dd->Union(*vec_Xd_in(),
                                 /*call_filter*/ datastructures::func_false);
  assert(Xd_nodes.size() == vec_Xd_in()->container().size());

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
  auto nodes = vec_Xdd_in()->Bfs();
  std::sort(nodes.begin(), nodes.end(), [](auto n1, auto n2) {
    return std::abs(n1->value()) > std::abs(n2->value());
  });
  double sq_norm = 0.0;
  for (auto node : nodes) sq_norm += node->value() * node->value();
  double cur_sq_norm = 0.0;
  size_t last_idx = 0;
  for (; last_idx < nodes.size(); last_idx++) {
    cur_sq_norm += nodes[last_idx]->value() * nodes[last_idx]->value();
    if (cur_sq_norm >= theta_ * theta_ * sq_norm) break;
  }
  nodes.resize(last_idx + 1);
  return nodes;
}

void AdaptiveHeatEquation::Refine(
    const std::vector<DoubleNodeVector<ThreePointWaveletFn, HierarchicalBasisFn>
                          *> &nodes_to_add) {
  X_d_.ConformingRefinement(*vec_Xdd_in(), nodes_to_add);

  vec_Xd_in_ =
      std::make_shared<TypeXVector>(X_d_.template DeepCopy<TypeXVector>());
  TypeXVector vec_Xd_out_tmp = X_d_.template DeepCopy<TypeXVector>();
  vec_Xd_out_tmp += *vec_Xd_out_;
  *vec_Xd_out_ = std::move(vec_Xd_out_tmp);

  vec_Xdd_in_ = std::make_shared<TypeXVector>(
      GenerateXDeltaUnderscore(X_d_, saturation_layers_)
          .template DeepCopy<TypeXVector>());
  vec_Xdd_out_ = std::make_shared<TypeXVector>(
      vec_Xdd_in_->template DeepCopy<TypeXVector>());

  vec_Ydd_in_ = std::make_shared<TypeYVector>(
      GenerateYDelta(*vec_Xdd_in_).template DeepCopy<TypeYVector>());
  vec_Ydd_out_ = std::make_shared<TypeYVector>(
      vec_Ydd_in_->template DeepCopy<TypeYVector>());

  heat_d_dd_ = std::make_unique<HeatEquation>(vec_Xd_in_, vec_Xd_out_,
                                              vec_Ydd_in_, vec_Ydd_out_);
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
