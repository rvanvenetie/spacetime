#include "error_estimator.hpp"

#include "space/integration.hpp"

namespace applications::ErrorEstimator {
namespace {
double u0L2NormSquared(HeatEquation &heat,
                       LinearFormBase<ThreePointWaveletFn> &u0_lf) {
  auto u0_functional = u0_lf.SpaceLF().Functional();
  auto u0 = u0_functional->Function();
  double u0_norm_sq = 0.0;
  auto space_metaroot = heat.vec_X()->Project_1()->node()->vertex();
  assert(space_metaroot->is_metaroot());
  space::Element2D *elem_metaroot =
      space_metaroot->children()[0]->patch[0]->parents()[0];
  assert(elem_metaroot->is_metaroot());
  auto u0_sq = [&u0](double x, double y) { return u0(x, y) * u0(x, y); };
  for (auto &elem : elem_metaroot->children())
    u0_norm_sq += space::Integrate(u0_sq, *elem, 2 * u0_functional->Order());
  return u0_norm_sq;
}

void ApplyMeanZero(
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
}  // namespace

GlobalError ComputeGlobalError(const Eigen::VectorXd &g_min_Bu,
                               const Eigen::VectorXd &PY_g_min_Bu,
                               const Eigen::VectorXd &G_u_dd_dd,
                               const Eigen::VectorXd &u0, HeatEquation &heat,
                               const Eigen::VectorXd &u_dd_dd,
                               LinearFormBase<ThreePointWaveletFn> &u0_lf) {
  GlobalError error;
  double error_Yprime_sq = PY_g_min_Bu.dot(g_min_Bu);

  // Compute ||u_0 - u(0)||_L2^2 as ||u_0||^2 - 2<u_0, u(0)> + ||u(0)||^2.
  double u0_norm_sq = u0L2NormSquared(heat, u0_lf);
  double u0_gamma0_u_inp = u0.dot(u_dd_dd);
  double gamma0_u_norm_sq = G_u_dd_dd.dot(u_dd_dd);
  double error_t0_sq = u0_norm_sq - 2 * u0_gamma0_u_inp + gamma0_u_norm_sq;

  error.error = sqrt(error_Yprime_sq + error_t0_sq);
  error.error_Yprime = sqrt(error_Yprime_sq);
  error.error_t0 = sqrt(error_t0_sq);
  return error;
}

double ComputeLocalErrors(
    DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> *residual_dd_dd,
    bool mean_zero) {
  if (mean_zero) ApplyMeanZero(residual_dd_dd);
  // Do a basis transformation for calculation of the residual wrt Psi_delta.
  double sq_norm = 0.0;
  for (auto &dblnode : residual_dd_dd->container()) {
    if (dblnode.is_metaroot()) continue;
    int lvl_diff = std::get<0>(dblnode.nodes())->level() -
                   std::get<1>(dblnode.nodes())->level();
    dblnode.set_value(dblnode.value() / sqrt(1.0 + pow(4.0, lvl_diff)));
    sq_norm += dblnode.value() * dblnode.value();
  }
  return sqrt(sq_norm);
}

}  // namespace applications::ErrorEstimator
