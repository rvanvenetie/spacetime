#pragma once

namespace applications {
template <typename TypeGLinForm, typename TypeU0LinForm>
AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::AdaptiveHeatEquation(
    TypeXDelta &&X_delta, TypeGLinForm &&g_lin_form,
    TypeU0LinForm &&u0_lin_form, double theta, size_t saturation_layers)
    : X_delta_(std::move(X_delta)),
      X_delta_underscore_(
          GenerateXDeltaUnderscore(X_delta_, saturation_layers)),
      Y_delta_underscore_(GenerateYDelta(X_delta_underscore_)),
      heat_eq_(X_delta_, Y_delta_underscore_),
      g_lin_form_(std::move(g_lin_form)),
      u0_lin_form_(std::move(u0_lin_form)),
      theta_(theta),
      saturation_layers_(saturation_layers) {}

template <typename TypeGLinForm, typename TypeU0LinForm>
Eigen::VectorXd AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::Solve(
    const Eigen::VectorXd &x0) {
  g_lin_form_.Apply(heat_eq_.vec_Y_in());
  u0_lin_form_.Apply(heat_eq_.vec_X_in());
  heat_eq_.B()->Apply();  // This is actually only needed to initialize BT()
  heat_eq_.Ainv()->Apply();
  heat_eq_.BT()->Apply();

  // Turn this into an eigen-friendly vector.
  Eigen::VectorXd rhs = heat_eq_.vec_X_in()->ToVectorContainer() +
                        heat_eq_.vec_X_out()->ToVectorContainer();

  auto [result, data] = tools::linalg::PCG(*heat_eq_.SchurMat(), rhs,
                                           *heat_eq_.PrecondX(), x0, 100, 1e-5);
  heat_eq_.vec_X_out()->FromVectorContainer(result);
  return result;
}

template <typename TypeGLinForm, typename TypeU0LinForm>
Eigen::VectorXd AdaptiveHeatEquation<TypeGLinForm, TypeU0LinForm>::Estimate(
    bool mean_zero) {
  HeatEquation heat_dd_dd(X_delta_underscore_, Y_delta_underscore_);
  auto u_dd_dd = heat_dd_dd.vec_X_in();
  *u_dd_dd += *heat_eq_.vec_X_out();

  // TODO: this code should not be copied.
  g_lin_form_.Apply(heat_dd_dd.vec_Y_in());
  u0_lin_form_.Apply(heat_dd_dd.vec_X_in());
  heat_dd_dd.B()->Apply();  // This is actually only needed to initialize BT()
  heat_dd_dd.Ainv()->Apply();
  heat_dd_dd.BT()->Apply();

  // Turn this into an eigen-friendly vector.
  Eigen::VectorXd rhs = heat_dd_dd.vec_X_in()->ToVectorContainer() +
                        heat_dd_dd.vec_X_out()->ToVectorContainer();
  rhs -= heat_dd_dd.SchurMat()->Apply();
  // Reuse u_dd_dd.
  u_dd_dd->FromVectorContainer(rhs);
  if (mean_zero) ApplyMeanZero(u_dd_dd);

  std::vector<DoubleNodeVector<ThreePointWaveletFn, HierarchicalBasisFn> *>
      X_d_nodes;
  u_dd_dd->Union(*heat_eq_.vec_X_in(),
                 /*call_filter*/ datastructures::func_false,
                 /* call_postprocess */
                 [&X_d_nodes](auto my_node, auto its_node) {
                   X_d_nodes.push_back(my_node);
                 });
  assert(X_d_nodes.size() == X_delta_.container().size());

  for (auto dblnode : X_d_nodes) dblnode->set_marked(true);
  for (auto &dblnode : u_dd_dd->container()) {
    if (dblnode.is_metaroot()) continue;
    if (dblnode.marked()) continue;
    int lvl_diff = std::get<0>(dblnode.nodes())->level() -
                   std::get<1>(dblnode.nodes())->level();
    dblnode.set_value(dblnode.value() / sqrt(1.0 + pow(4.0, lvl_diff)));
  }

  for (auto dblnode : X_d_nodes) dblnode->set_marked(false);

  return u_dd_dd->ToVectorContainer();
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
