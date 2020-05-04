#include <boost/range/adaptor/reversed.hpp>

#include "adaptive_heat_equation.hpp"
namespace applications {

class XEquivalentErrorEstimator {
 public:
  static double ComputeGlobalError(
      HeatEquation &heat, AdaptiveHeatEquation::TypeYLinForm &g_lin_form,
      AdaptiveHeatEquation::TypeXLinForm &u0_lin_form,
      const Eigen::VectorXd &u_dd_d) {
    Eigen::VectorXd Bu_min_g =
        heat.B()->Apply(u_dd_d) - g_lin_form.Apply(heat.vec_Y());
    auto [Ainv_Bu_min_g, _] =
        tools::linalg::PCG(*heat.A(), Bu_min_g, *heat.P_Y(),
                           Eigen::VectorXd::Zero(Bu_min_g.rows()),
                           /*imax*/ 100, /*rtol*/ 1e-5);
    double residual_norm = Ainv_Bu_min_g.dot(Bu_min_g);
    std::cout << std::endl << residual_norm << std::endl;
    return residual_norm;
  }
};

class ResidualErrorEstimator {
 public:
  static double ComputeLocalErrors(
      AdaptiveHeatEquation::TypeXVector *residual_dd_dd,
      bool mean_zero = true) {
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

  static void ApplyMeanZero(AdaptiveHeatEquation::TypeXVector *vec) {
    for (auto &dblnode : boost::adaptors::reverse(vec->container())) {
      auto [_, space_node] = dblnode.nodes();
      if (space_node->level() == 0 || space_node->on_domain_boundary())
        continue;
      if (std::any_of(space_node->parents().begin(),
                      space_node->parents().end(),
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
};
}  // namespace applications
