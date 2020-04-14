#include "heat_equation.hpp"
namespace applications {
using spacetime::GenerateYDelta;

template <bool UseCache>
HeatEquation<UseCache>::HeatEquation(std::shared_ptr<TypeXVector> vec_X_in,
                                     std::shared_ptr<TypeXVector> vec_X_out,
                                     std::shared_ptr<TypeYVector> vec_Y_in,
                                     std::shared_ptr<TypeYVector> vec_Y_out,
                                     std::shared_ptr<TypeA> A,
                                     std::shared_ptr<TypeAinv> Ainv)
    : vec_X_in_(vec_X_in),
      vec_X_out_(vec_X_out),
      vec_Y_in_(vec_Y_in),
      vec_Y_out_(vec_Y_out),
      A_(A),
      A_inv_(Ainv) {
  // Create two parts of B sharing sigma and theta.
  auto B_t = std::make_shared<TypeB_t>(vec_X_in_.get(), vec_Y_out_.get());
  auto B_s = std::make_shared<TypeB_s>(vec_X_in_.get(), vec_Y_out_.get(),
                                       B_t->sigma(), B_t->theta());
  B_ = std::make_shared<TypeB>(B_t, B_s);

  // Create transpose of B sharing data with B.
  InitializeBT();

  // Create trace operator.
  G_ = std::make_shared<TypeG>(vec_X_in_.get(), vec_X_out_.get());

  // Create the negative trace operator.
  auto minus_G = std::make_shared<NegativeBilinearForm<TypeG>>(G_);

  // Create the block matrix.
  block_bf_ = std::make_shared<TypeBlockBF>(A_, B_, BT_, minus_G);
  // Create the Schur matrix.
  schur_bf_ = std::make_shared<TypeSchurBF>(A_inv_, B_, BT_, G_);
  precond_X_ =
      std::make_shared<TypePrecondX>(vec_X_in_.get(), vec_X_out_.get());
}

template <bool UseCache>
HeatEquation<UseCache>::HeatEquation(std::shared_ptr<TypeXVector> vec_X_in,
                                     std::shared_ptr<TypeXVector> vec_X_out,
                                     std::shared_ptr<TypeYVector> vec_Y_in,
                                     std::shared_ptr<TypeYVector> vec_Y_out)
    : HeatEquation<UseCache>(
          vec_X_in, vec_X_out, vec_Y_in, vec_Y_out,
          std::make_shared<TypeA>(vec_Y_in.get(), vec_Y_out.get(), UseCache),
          std::make_shared<TypeAinv>(vec_Y_out.get(), vec_Y_in.get(),
                                     UseCache)) {}

template <bool UseCache>
HeatEquation<UseCache>::HeatEquation(const TypeXDelta &X_delta,
                                     const TypeYDelta &Y_delta)
    : HeatEquation<UseCache>(std::make_shared<TypeXVector>(
                                 X_delta.template DeepCopy<TypeXVector>()),
                             std::make_shared<TypeXVector>(
                                 X_delta.template DeepCopy<TypeXVector>()),
                             std::make_shared<TypeYVector>(
                                 Y_delta.template DeepCopy<TypeYVector>()),
                             std::make_shared<TypeYVector>(
                                 Y_delta.template DeepCopy<TypeYVector>())) {}

template <bool UseCache>
HeatEquation<UseCache>::HeatEquation(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta)
    : HeatEquation<UseCache>(X_delta, GenerateYDelta(X_delta)) {}

template <>
void HeatEquation<true>::InitializeBT() {
  BT_ = std::make_shared<TypeBT>(B_->Transpose(), vec_Y_in_.get(),
                                 vec_X_out_.get());

  // Create trace operator.
  G_ = std::make_shared<TypeG>(vec_X_in_.get(), vec_X_out_.get());

  // Create the negative trace operator.
  auto minus_G = std::make_shared<NegativeBilinearForm<TypeG>>(G_);

  // Create the block bilinear form.
  block_bf_ = std::make_shared<TypeBlockBF>(A_, B_, BT_, minus_G);
  // Create the Schur bilinear form.
  schur_bf_ = std::make_shared<TypeSchurBF>(A_inv_, B_, BT_, G_);
  precond_X_ =
      std::make_shared<TypePrecondX>(vec_X_in_.get(), vec_X_out_.get());
}

template <>
void HeatEquation<false>::InitializeBT() {
  auto BT_t = std::make_shared<
      BilinearForm<Time::TransportOperator, space::MassOperator,
                   OrthonormalWaveletFn, ThreePointWaveletFn>>(
      vec_Y_in_.get(), vec_X_out_.get(), B_->theta(), B_->sigma());
  auto BT_s = std::make_shared<
      BilinearForm<Time::MassOperator, space::StiffnessOperator,
                   OrthonormalWaveletFn, ThreePointWaveletFn>>(
      vec_Y_in_.get(), vec_X_out_.get(), B_->theta(), B_->sigma());
  BT_ = std::make_shared<TypeBT>(BT_t, BT_s);
}

template class HeatEquation<false>;
template class HeatEquation<true>;
};  // namespace applications
