#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace tools::linalg {

struct SolverData {
  double relative_residual;
  size_t iterations;
  bool converged;
};

// Loosely based off Eigen/ConjugateGradient.h.
template <typename MatType, typename PrecondType>
std::pair<Eigen::VectorXd, SolverData> PCG(const MatType &A,
                                           const Eigen::VectorXd &b,
                                           const PrecondType &M,
                                           const Eigen::VectorXd &x0, int imax,
                                           double atol);

template <typename MatType, typename PrecondType>
class Lanczos {
 public:
  Lanczos(const MatType &A, const PrecondType &P, size_t max_iterations = 200,
          double tol = 0.0001, double tol_bisec = 0.000001)
      : Lanczos(A, P, Eigen::VectorXd::Random(A.cols()), max_iterations, tol,
                tol_bisec) {}
  Lanczos(const MatType &A, const PrecondType &P,
          const Eigen::VectorXd &initial_guess, size_t max_iterations = 200,
          double tol = 0.0001, double tol_bisec = 0.000001);

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

  void bisec(size_t k, double &ymax, double &zmin, double tol_bisec);
  double pol(int k, double x);
};

};  // namespace tools::linalg

#include "linalg.ipp"
