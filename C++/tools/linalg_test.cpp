#include "linalg.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tools {
TEST(PCG, CanSolve) {
  // Example from Wikipedia.
  Eigen::VectorXd b(2), zero(2), x0(2), x_true(2);
  b << 1, 2;
  zero << 0, 0;
  x0 << 2, 1;
  x_true << 1.0 / 11, 7.0 / 11;
  Eigen::MatrixXd A(2, 2), Id(2, 2), M(2, 2);
  A << 4, 1, 1, 3;
  Id << 1, 0, 0, 1;
  M << 4, 0, 0, 3;

  ASSERT_TRUE(linalg::PCG(A, b, Id, zero, 10, 1e-5).first.isApprox(x_true));
  ASSERT_TRUE(linalg::PCG(A, b, M, zero, 10, 1e-5).first.isApprox(x_true));
  ASSERT_TRUE(linalg::PCG(A, b, Id, x0, 10, 1e-5).first.isApprox(x_true));
  ASSERT_TRUE(linalg::PCG(A, b, M, x0, 10, 1e-5).first.isApprox(x_true));
}

TEST(Lanczos, identity) {
  auto id = Eigen::Matrix<double, 10, 10>::Identity();

  linalg::Lanczos lanczos(id, id);

  ASSERT_DOUBLE_EQ(lanczos.min(), 1.0);
  ASSERT_DOUBLE_EQ(lanczos.max(), 1.0);
}

TEST(Lanczos, diagonal) {
  auto id = Eigen::Matrix<double, 10, 10>::Identity();

  {
    Eigen::Matrix<double, 10, 1> vec;
    vec << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    linalg::Lanczos lanczos(vec.asDiagonal(), id);
    ASSERT_NEAR(lanczos.min(), 1.0, 1e-3);
    ASSERT_NEAR(lanczos.max(), 10.0, 1e-2);
  }

  {
    Eigen::Matrix<double, 10, 1> vec;
    vec << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1000;

    linalg::Lanczos lanczos(vec.asDiagonal(), id);
    ASSERT_NEAR(lanczos.min(), 0.1, 1e-4);
    ASSERT_NEAR(lanczos.max(), 1000.0, 1);
  }

  {
    Eigen::Matrix<double, 10, 1> vec;
    for (int i = -5; i < 5; ++i) vec[i + 5] = pow(10, i);

    linalg::Lanczos lanczos(vec.asDiagonal(), id);
    ASSERT_NEAR(lanczos.max(), pow(10, 4), 1);
  }
}

TEST(Lanczos, RandomMatrix) {
  for (size_t i = 1; i < 15; ++i) {
    auto id = Eigen::MatrixXd::Identity(i, i);

    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(i, i);
    Eigen::MatrixXd Q = mat.colPivHouseholderQr().matrixQ();
    Eigen::VectorXd diag(i);

    for (size_t j = 0; j < i; ++j) diag[j] = pow(10, -2);

    diag[0] = 100;

    Eigen::MatrixXd A = Q * diag.asDiagonal() * Q.transpose();

    linalg::Lanczos lanczos(A, id);
    ASSERT_NEAR(lanczos.min(), diag.minCoeff(), 1e-3);
    ASSERT_NEAR(lanczos.max(), diag.maxCoeff(), 1e-1);
  }
}

TEST(Lanczos, PreconditionedRandom) {
  for (size_t i = 1; i < 15; ++i) {
    auto id = Eigen::MatrixXd::Identity(i, i);

    // Random orthonormal matrix.
    Eigen::MatrixXd Q_A =
        Eigen::MatrixXd::Random(i, i).colPivHouseholderQr().matrixQ();
    Eigen::MatrixXd Q_P =
        Eigen::MatrixXd::Random(i, i).colPivHouseholderQr().matrixQ();

    // Random diagonal with values in [1,2].
    Eigen::VectorXd D_A = Eigen::VectorXd::Random(i).cwiseAbs();
    Eigen::VectorXd D_P = Eigen::VectorXd::Random(i).cwiseAbs();
    D_A += Eigen::VectorXd::Ones(i);
    D_P += Eigen::VectorXd::Ones(i);

    // SPD matrices with eigenvalues in [1,2].
    Eigen::MatrixXd A = Q_A * D_A.asDiagonal() * Q_A.transpose();
    Eigen::MatrixXd P = Q_P * D_P.asDiagonal() * Q_P.transpose();

    // Compare Lanczos(A,P) to the matrix PA.
    Eigen::MatrixXd PA = P * A;
    Eigen::VectorXd PA_eig = PA.eigenvalues().real();

    ASSERT_NEAR(lanczos_A_P.min(), PA_eig.minCoeff(), 1e-3);
    ASSERT_NEAR(lanczos_A_P.max(), PA_eig.maxCoeff(), 1e-3);
  }
}

}  // namespace tools
