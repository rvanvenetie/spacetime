#include "heat_equation.hpp"

using spacetime::GenerateYDelta;

namespace applications {
HeatEquation::HeatEquation(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta,
    const DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn> &Y_delta)
    : vec_X_in_(X_delta.template DeepCopy<
                DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>()),
      vec_X_out_(X_delta.template DeepCopy<
                 DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>()),
      vec_Y_in_(Y_delta.template DeepCopy<
                DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>()),
      vec_Y_out_(
          Y_delta.template DeepCopy<
              DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>()) {
  // Create A_s mapping from Y_delta to Y_delta.
  A_ = std::make_shared<TypeA>(&vec_Y_in_, &vec_Y_out_);

  // Create two parts of B sharing sigma and theta.
  auto B_t = std::make_shared<TypeB_t>(&vec_X_in_, &vec_Y_out_);
  auto B_s = std::make_shared<TypeB_s>(&vec_X_in_, &vec_Y_out_, B_t->sigma(),
                                       B_t->theta());
  B_ = std::make_shared<TypeB>(B_t, B_s);

  // Create transpose of B sharing data with B.
  BT_ = std::make_shared<TypeBT>(B_->Transpose(), &vec_Y_in_, &vec_X_out_);

  // Create trace operator.
  G_ = std::make_shared<TypeG>(&vec_X_in_, &vec_X_out_);

  // Create the negative trace operator.
  auto minus_G = std::make_shared<NegativeBilinearForm<TypeG>>(G_);

  // Craete the block matrix.
  block_mat_ = std::make_shared<TypeBlockMat>(A_, B_, BT_, minus_G);
}

HeatEquation::HeatEquation(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta)
    : HeatEquation(X_delta, GenerateYDelta(X_delta)) {}

}  // namespace applications
