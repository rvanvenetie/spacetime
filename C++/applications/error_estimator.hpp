#include <boost/range/adaptor/reversed.hpp>

#include "adaptive_heat_equation.hpp"
#include "space/integration.hpp"
namespace applications {

class XEquivalentErrorEstimator {
 public:
  static double ComputeGlobalError(HeatEquation &heat,
                                   AdaptiveHeatEquation::TypeYLinForm &g_lf,
                                   AdaptiveHeatEquation::TypeXLinForm &u0_lf,
                                   const Eigen::VectorXd &u_dd_d) {
    // Compute ||Bu - g||_Y^2.
    Eigen::VectorXd Bu_min_g =
        heat.B()->Apply(u_dd_d) - g_lf.Apply(heat.vec_Y());
    auto [Ainv_Bu_min_g, _] =
        tools::linalg::PCG(*heat.A(), Bu_min_g, *heat.P_Y(),
                           Eigen::VectorXd::Zero(Bu_min_g.rows()),
                           /*imax*/ 100, /*rtol*/ 1e-5);
    double residual_Ynorm_sq = Ainv_Bu_min_g.dot(Bu_min_g);

    // Compute ||u_0 - u(0)||_L2^2 as ||u_0||^2 - 2<u_0, u(0)> + ||u(0)||^2.
    double u0_norm_sq = u0L2NormSquared(heat, u0_lf);
    double u0_gamma0_u_inp = u_dd_d.dot(u0_lf.Apply(heat.vec_X()));
    double gamma0_u_norm_sq = u_dd_d.dot(heat.G()->Apply(u_dd_d));
    double u_error_sq = u0_norm_sq - 2 * u0_gamma0_u_inp + gamma0_u_norm_sq;
    return sqrt(residual_Ynorm_sq + u_error_sq);
  }

 protected:
  static double u0L2NormSquared(HeatEquation &heat,
                                AdaptiveHeatEquation::TypeXLinForm &u0_lf) {
    auto u0_functional = static_cast<space::QuadratureFunctional *>(
        static_cast<spacetime::LinearForm<Time::ThreePointWaveletFn> &>(u0_lf)
            .SpaceLF()
            .Functional());
    auto u0 = u0_functional->Function();
    double u0_norm_sq = 0.0;
    for (auto &[elem, _vids] :
         space::TriangulationView(heat.vec_X()->Project_1()->Bfs())
             .InitialTriangulationView()
             .element_leaves())
      u0_norm_sq += space::Integrate(
          [&u0](double x, double y) { return u0(x, y) * u0(x, y); }, *elem,
          2 * u0_functional->Order());
    return u0_norm_sq;
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
