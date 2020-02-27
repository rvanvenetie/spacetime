#include <Eigen/Dense>
#include <vector>

namespace tools::linalg {

template <typename MatType, typename PrecondType>
std::pair<Eigen::VectorXd, std::pair<double, int>> PCG(
    const MatType &A, const Eigen::VectorXd &b, const PrecondType &M,
    const Eigen::VectorXd &x0, const int jmax, const double rtol) {
  assert(A.rows() == A.cols());
  const int n = A.rows();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

  Eigen::VectorXd residual = b - A * x0;
  double sq_rhs_norm = b.squaredNorm();
  if (sq_rhs_norm == 0) return {x, {0.0, 1}};

  double threshold = rtol * rtol * sq_rhs_norm;
  double sq_res_norm = residual.squaredNorm();
  if (sq_rhs_norm == 0) {
    x += x0;
    return {x, {0.0, 1}};
  }

  Eigen::VectorXd p = M * residual;
  Eigen::VectorXd z(n), tmp(n);
  double abs_r = residual.dot(p);

  size_t i = 0;
  while (i < jmax) {
    tmp.noalias() = A * p;
    double alpha = abs_r / p.dot(tmp);
    x += alpha * p;
    residual -= alpha * tmp;
    sq_res_norm = residual.squaredNorm();
    if (sq_res_norm < threshold) break;

    z = M * residual;
    double abs_r_old = abs_r;
    abs_r = residual.dot(z);
    double beta = abs_r / abs_r_old;
    p = z + beta * p;

    i++;
  }

  return {x, {sqrt(sq_res_norm / sq_rhs_norm), i}};
}

};  // namespace tools::linalg
