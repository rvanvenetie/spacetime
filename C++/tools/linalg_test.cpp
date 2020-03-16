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
}  // namespace tools
