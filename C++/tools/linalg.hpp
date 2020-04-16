#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <vector>

namespace tools::linalg {

// Loosely based off Eigen/ConjugateGradient.h.
template <typename MatType, typename PrecondType>
std::pair<Eigen::VectorXd, std::pair<double, int>> PCG(
    const MatType &A, const Eigen::VectorXd &b, const PrecondType &M,
    const Eigen::VectorXd &x0, int imax, double rtol) {
  assert(A.rows() == A.cols());
  const int n = A.rows();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

  double sq_rhs_norm = b.squaredNorm();
  if (sq_rhs_norm == 0) return {x, {0.0, 1}};

  double threshold = rtol * rtol * sq_rhs_norm;
  Eigen::VectorXd residual = b - A * x0;
  double sq_res_norm = residual.squaredNorm();
  if (sq_rhs_norm == 0) return {x0, {0.0, 1}};
  x = x0;

  Eigen::VectorXd p = M * residual;
  Eigen::VectorXd z(n), tmp(n);
  double abs_r = residual.dot(p);

  size_t i = 0;
  while (i < imax) {
    tmp.noalias() = A * p;
    double alpha = abs_r / p.dot(tmp);
    x += alpha * p;
    residual -= alpha * tmp;
    sq_res_norm = residual.squaredNorm();
    if (sq_res_norm < threshold) break;

    z.noalias() = M * residual;
    double abs_r_old = abs_r;
    abs_r = residual.dot(z);
    double beta = abs_r / abs_r_old;
    p = z + beta * p;

    i++;
  }
  if (sq_res_norm > threshold) std::cout << "DID NOT CONVERGE" << std::endl;

  return {x, {sqrt(sq_res_norm / sq_rhs_norm), i}};
}

template <typename MatType, typename PrecondType>
class Lanczos {
 public:
  Lanczos(const MatType &A, const PrecondType &P, size_t max_iterations = 200,
          double tol = 0.0001, double tol_bisec = 0.000001)
      : Lanczos(A, P, Eigen::VectorXd::Random(A.cols()), max_iterations, tol,
                tol_bisec) {}
  Lanczos(const MatType &A, const PrecondType &P,
          const Eigen::VectorXd &initial_guess, size_t max_iterations = 200,
          double tol = 0.0001, double tol_bisec = 0.000001)
      : alpha_(max_iterations), beta_(max_iterations - 1), converged_(true) {
    assert(P.cols() == A.rows());
    using clock_t = std::chrono::steady_clock;
    auto start = clock_t::now();

    double lmaxold, lminold;

    // start with random initial guess
    Eigen::VectorXd w = initial_guess;
    Eigen::VectorXd v(A.rows());
    Eigen::VectorXd u(A.rows());

    v.noalias() = A * w;  // v=Aw

    double norm = sqrt(v.dot(w));

    v /= norm;
    w /= norm;

    v = P * v;  // v=Pv
    u = A * v;  // u=Av

    alpha_[0] = u.dot(w);

    lmax_ = lmin_ = alpha_[0];

    size_t k = 0;
    do {
      if (k == max_iterations - 1) {
        converged_ = false;
        break;
      }

      v -= alpha_[k] * w;  // v=v-alpha_[k]w
      u = A * v;
      beta_[k] = sqrt(u.dot(v));  // beta_[k]=||v||
      if (beta_[k] == 0) break;

      Eigen::VectorXd temp = w;

      w = v / beta_[k];
      v = -beta_[k] * temp;

      u = A * w;
      u = P * u;
      v += u;

      u = A * v;
      alpha_[++k] = u.dot(w);  // alpha_[++k]=<v,w>

      lmaxold = lmax_, lminold = lmin_;

      bisec(k, lmax_, lmin_, tol_bisec);

    } while ((lmax_ - lmaxold) > tol * lmaxold ||
             (lminold - lmin_) > tol * lmin_);
    iterations_ = k + 1;
    alpha_.conservativeResize(k);
    if (k) beta_.conservativeResize(k - 1);

    time_ = std::chrono::duration<float>(clock_t::now() - start).count();
  }

  double max() const { return lmax_; }
  double min() const { return lmin_; }
  double cond() const { return lmax_ / lmin_; }
  float time() const { return time_; };

  size_t iterations() const { return iterations_; }
  bool converged() const { return converged_; }

  // overload the << operator
  friend std::ostream &operator<<(std::ostream &os, const Lanczos &lanczos) {
    if (lanczos.converged())
      os << "converged\t";
    else
      os << "NOT converged\t";

    os << "its=" << lanczos.iterations() << "\tlmax=" << lanczos.max()
       << "\tlmin=" << lanczos.min() << "\tkappa=" << lanczos.cond()
       << "\ttime=" << lanczos.time() << " s";
    return os;
  }

 private:
  Eigen::VectorXd alpha_, beta_;
  double lmax_, lmin_;

  size_t iterations_;
  bool converged_;
  float time_;

  void bisec(size_t k, double &ymax, double &zmin, double tol_bisec) {
    // precondition: 0 < (ymax-lmax[k-1])/ymax < TOLBISEC,
    //               0 < (lmin[k-1]-zmin)/zmin < TOLBISEC

    double zmax, ymin, x, px, pz, py;
    size_t l;

    zmax = alpha_[0] + fabs(beta_[0]);
    ymin = alpha_[0] - fabs(beta_[0]);
    for (l = 1; l < k; l++) {
      zmax = std::max(zmax, alpha_[l] + fabs(beta_[l - 1]) + fabs(beta_[l]));
      ymin = std::min(ymin, alpha_[l] - fabs(beta_[l - 1]) - fabs(beta_[l]));
    }
    zmax = std::max(zmax, alpha_[k] + fabs(beta_[k - 1]));  // upp for lmax[k]
    ymin = std::min(ymin, alpha_[k] - fabs(beta_[k - 1]));  // low for lmin[k]
    ymin = std::max(ymin,
                    0.0);  // ymin >= 0 because we are dealing with spd matrices

    pz = pol(k, zmax);
    while (fabs(zmax - ymax) > tol_bisec * std::min(fabs(zmax), fabs(ymax))) {
      x = (ymax + zmax) / 2;
      px = pol(k, x);
      if (std::signbit(px) != std::signbit(pz))  // lmax[k] in [x,zmax]
      {
        ymax = x;
      } else  // lmax[k]<x
      {
        zmax = x;
        pz = px;
      }
    }
    py = pol(k, ymax);
    if (std::signbit(pz) != std::signbit(py) &&
        py != 0)  // lmax[k] in (ymax,zmax]
      ymax = zmax;

    // postcondition: 0 < (ymax-lmax[k])/ymax < TOLBISEC

    py = pol(k, ymin);
    while (fabs(zmin - ymin) > tol_bisec * std::min(fabs(zmin), fabs(ymin))) {
      x = (ymin + zmin) / 2;
      px = pol(k, x);

      if (std::signbit(px) != std::signbit(py))  // lmin[k] in [ymin,x]
      {
        zmin = x;
      } else  // lmin[k]>x
      {
        ymin = x;
        py = px;
      }
    }
    pz = pol(k, zmin);
    if (std::signbit(pz) != std::signbit(py) &&
        pz != 0)  // lmin[k] in [ymin,zmin)
      zmin = ymin;

    // postcondition: 0 < (lmin[k-1]-zmin)/zmin < TOLBISEC
  }

  double pol(int k, double x) {
    double p, q, r;
    int l;
    r = 1;
    p = alpha_[0] - x;
    for (l = 1; l < k + 1; l++) {
      q = p;
      p = (alpha_[l] - x) * p - beta_[l - 1] * beta_[l - 1] * r;
      r = q;
    }
    return p;
  }
};

};  // namespace tools::linalg
