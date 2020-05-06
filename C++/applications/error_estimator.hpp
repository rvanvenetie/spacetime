#pragma once
#include <boost/range/adaptor/reversed.hpp>

#include "adaptive_heat_equation.hpp"
namespace applications {
namespace ErrorEstimator {
double ComputeGlobalError(HeatEquation &heat,
                          AdaptiveHeatEquation::TypeYLinForm &g_lf,
                          AdaptiveHeatEquation::TypeXLinForm &u0_lf,
                          const Eigen::VectorXd &u_dd_d);

double ComputeLocalErrors(AdaptiveHeatEquation::TypeXVector *residual_dd_dd,
                          bool mean_zero = true);
}  // namespace ErrorEstimator
}  // namespace applications
