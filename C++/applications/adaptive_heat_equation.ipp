#pragma once
#include <boost/range/adaptor/reversed.hpp>

namespace applications {
template <typename TypeGLinForm, typename TypeU0LinForm>
AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::AdaptiveHeatEquation(
    TypeXDelta &&X_delta, TypeGLinForm &&g_lin_form,
    TypeU0LinForm &&u0_lin_form, double theta, size_t saturation_layers)
    : X_d_(std::move(X_delta)),
      X_dd_(GenerateXDeltaUnderscore(X_d_, saturation_layers)),
      vec_Xd_in_(
          std::make_shared<TypeXVector>(X_d_.template DeepCopy<TypeXVector>())),
      vec_Xd_out_(
          std::make_shared<TypeXVector>(X_d_.template DeepCopy<TypeXVector>())),
      vec_Xdd_in_(std::make_shared<TypeXVector>(
          X_dd_.template DeepCopy<TypeXVector>())),
      vec_Xdd_out_(std::make_shared<TypeXVector>(
          X_dd_.template DeepCopy<TypeXVector>())),
      Y_dd_(GenerateYDelta(X_dd_)),
      vec_Ydd_in_(std::make_shared<TypeYVector>(
          Y_dd_.template DeepCopy<TypeYVector>())),
      vec_Ydd_out_(std::make_shared<TypeYVector>(
          Y_dd_.template DeepCopy<TypeYVector>())),
      heat_d_dd_(vec_Xd_in_, vec_Xd_out_, vec_Ydd_in_, vec_Ydd_out_),
      heat_dd_dd_(vec_Xdd_in_, vec_Xdd_out_, vec_Ydd_in_, vec_Ydd_out_),
      g_lin_form_(std::move(g_lin_form)),
      u0_lin_form_(std::move(u0_lin_form)),
      theta_(theta),
      saturation_layers_(saturation_layers) {}

template <typename TypeGLinForm, typename TypeU0LinForm>
Eigen::VectorXd AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::RHS(
    HeatEquation &heat) {
  heat.B()->Apply();  // This is actually only needed to initialize BT()
  g_lin_form_.Apply(heat.vec_Y_out());
  heat.Ainv()->Apply();
  auto rhs = heat.BT()->Apply();
  rhs += u0_lin_form_.Apply(heat.vec_X_in());
  return rhs;
}

template <typename TypeGLinForm, typename TypeU0LinForm>
DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>
    *AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::Solve(
        const Eigen::VectorXd &x0, double rtol, size_t maxit) {
  auto rhs = RHS(heat_d_dd_);
  auto [result, data] = tools::linalg::PCG(
      *heat_d_dd_.SchurMat(), rhs, *heat_d_dd_.PrecondX(), x0, maxit, rtol);
  vec_Xd_out()->FromVectorContainer(result);
  return vec_Xd_out();
}

template <typename TypeGLinForm, typename TypeU0LinForm>
DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> *
AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::Estimate(bool mean_zero) {
  auto u_dd_dd = vec_Xdd_in();
  u_dd_dd->Reset();
  *u_dd_dd += *vec_Xd_out();
  auto Su_dd_dd = heat_dd_dd_.SchurMat()->Apply();

  // Reuse u_dd_dd.
  u_dd_dd->FromVectorContainer(RHS(heat_dd_dd_) - Su_dd_dd);
  if (mean_zero) ApplyMeanZero(u_dd_dd);

  auto X_d_nodes = u_dd_dd->Union(*vec_Xd_in(),
                                  /*call_filter*/ datastructures::func_false);
  assert(X_d_nodes.size() == X_d_.container().size());

  for (auto dblnode : X_d_nodes) dblnode->set_marked(true);
  for (auto &dblnode : u_dd_dd->container()) {
    if (dblnode.is_metaroot()) continue;
    if (dblnode.marked()) continue;
    int lvl_diff = std::get<0>(dblnode.nodes())->level() -
                   std::get<1>(dblnode.nodes())->level();
    dblnode.set_value(dblnode.value() / sqrt(1.0 + pow(4.0, lvl_diff)));
  }

  for (auto dblnode : X_d_nodes) dblnode->set_marked(false);

  return u_dd_dd;
}

template <typename TypeGLinForm, typename TypeU0LinForm>
void AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::Mark() {
  auto nodes = vec_Xdd_in()->Bfs();
  std::sort(nodes.begin(), nodes.end(), [](auto n1, auto n2) {
    return std::abs(n1->value()) > std::abs(n2->value());
  });
  double sq_norm = 0.0;
  for (auto node : nodes) sq_norm += node->value() * node->value();
  double cur_sq_norm = 0.0;
  for (size_t i = 0; i < nodes.size(); i++) {
    cur_sq_norm += nodes[i]->value() * nodes[i]->value();
    nodes[i]->set_marked(true);
    if (cur_sq_norm < theta_ * theta_ * sq_norm) break;
  }
  auto X_d_nodes =
      vec_Xdd_in()->Union(*vec_Xd_in(),
                          /*call_filter*/ datastructures::func_false);
  for (auto node : X_d_nodes) node->set_marked(true);
  return;
}

template <typename TypeGLinForm, typename TypeU0LinForm>
void AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::Refine() {
  // X_d_ = std::move(vec_Xdd_in().template MakeConforming<TypeXDelta>());
  // X_dd_ = std::move(GenerateXDeltaUnderscore(X_d_, saturation_layers_));
  // Y_dd_ = std::move(GenerateYDelta(X_dd_));
  // heat_d_dd_ = std::move(HeatEquation(X_d_, Y_dd_));
  // heat_dd_dd_ = std::move(
  //     HeatEquation(X_dd_, Y_dd_));  // TODO: re-use
  //     heat_d_dd_.vec_Y_{in,out}.
}

template <typename TypeGLinForm, typename TypeU0LinForm>
void AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::ApplyMeanZero(
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
