#include <Eigen/Dense>
#include <vector>

namespace tools {
namespace linalg {

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

template <typename MatType, typename PrecondType>
std::pair<Eigen::VectorXd, std::vector<double>> PCG2(
    const MatType &A, const Eigen::VectorXd &b, const PrecondType &M,
    const Eigen::VectorXd &x0, const int jmax, const double rtol) {
  assert(A.rows() == A.cols());
  const int n = A.rows();
  Eigen::VectorXd r1(b);
  Eigen::VectorXd r0(b);
  Eigen::VectorXd p(Eigen::VectorXd::Zero(n));
  Eigen::VectorXd x(x0);
  Eigen::VectorXd z0 = M * r0;
  std::vector<double> res;
  res.reserve(jmax);
  double errInit = b.norm();

  for (size_t i = 0; i < jmax; i++) {
    Eigen::VectorXd z1 = M * r1;
    double r1z1 = r1.dot(z1);
    double mu = r1z1 / r0.dot(z0);
    z0 = z1;
    p = mu * p + z1;
    Eigen::VectorXd ap = A * p;
    double sigma = r1z1 / p.dot(ap);
    x = x + sigma * p;
    r0 = r1;
    r1 -= sigma * ap;
    double err = r1.norm();
    res.push_back(err);
    if (err <= rtol * errInit) break;
  }

  return {x, res};
}

};  // namespace linalg
};  // namespace tools
