#pragma once
#include <boost/range/adaptor/reversed.hpp>

#include "applications/heat_equation.hpp"
#include "datastructures/double_tree_view.hpp"
#include "spacetime/linear_form.hpp"

namespace applications {
namespace ErrorEstimator {
using datastructures::DoubleTreeVector;
using space::HierarchicalBasisFn;
using spacetime::LinearFormBase;
using Time::ThreePointWaveletFn;

struct GlobalError {
  double error;         // e_\delta
  double error_Yprime;  // \eqsim ||g - Bu||_{Y'}
  double error_t0;      // ||\gamma_0u - u0||_L2
};

GlobalError ComputeGlobalError(const Eigen::VectorXd &g_min_Bu,
                               const Eigen::VectorXd &PY_g_min_Bu,
                               HeatEquation &heat,
                               const Eigen::VectorXd &u_dd_dd,
                               LinearFormBase<ThreePointWaveletFn> &u0_lf);

double ComputeLocalErrors(
    DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> *residual_dd_dd,
    bool mean_zero = true);
}  // namespace ErrorEstimator
}  // namespace applications
