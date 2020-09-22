#pragma once
#include <boost/range/adaptor/reversed.hpp>

#include "adaptive_heat_equation.hpp"
namespace applications {
namespace ErrorEstimator {

struct GlobalError {
  double error;
  double error_Yprime;
  double error_t0;
};

GlobalError ComputeGlobalError(const Eigen::VectorXd &g_min_Bu,
                               const Eigen::VectorXd &PY_g_min_Bu,
                               HeatEquation &heat,
                               const Eigen::VectorXd &u_dd_dd,
                               AdaptiveHeatEquation::TypeXLinForm &u0_lf);

double ComputeLocalErrors(AdaptiveHeatEquation::TypeXVector *residual_dd_dd,
                          bool mean_zero = true);
}  // namespace ErrorEstimator
}  // namespace applications
