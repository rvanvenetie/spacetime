#include "heat_equation.hpp"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using spacetime::GenerateYDelta;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;
using TypeXDelta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>;
using TypeYDelta = DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>;
using TypeXVector = DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>;
using TypeYVector = DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>;

namespace applications {
HeatEquation::HeatEquation(std::shared_ptr<TypeXVector> vec_X_in,
                           std::shared_ptr<TypeXVector> vec_X_out,
                           std::shared_ptr<TypeYVector> vec_Y_in,
                           std::shared_ptr<TypeYVector> vec_Y_out,
                           std::shared_ptr<TypeA> A,
                           std::shared_ptr<TypePrecondY> P_Y,
                           const HeatEquationOptions &opts)
    : vec_X_in_(vec_X_in),
      vec_X_out_(vec_X_out),
      vec_Y_in_(vec_Y_in),
      vec_Y_out_(vec_Y_out),
      A_(A),
      P_Y_(P_Y),
      opts_(opts) {
  // Create two parts of B sharing sigma and theta.
  auto B_t = std::make_shared<TypeB_t>(vec_X_in_.get(), vec_Y_out_.get());
  auto B_s = std::make_shared<TypeB_s>(vec_X_in_.get(), vec_Y_out_.get(),
                                       B_t->sigma(), B_t->theta());
  B_ = std::make_shared<TypeB>(B_t, B_s);

  // Create transpose of B sharing data with B.
  BT_ = std::make_shared<TypeBT>(B_->Transpose(), vec_Y_in_.get(),
                                 vec_X_out_.get());

  // Create trace operator.
  G_ = std::make_shared<TypeG>(vec_X_in_.get(), vec_X_out_.get());

  // Create the negative trace operator.
  auto minus_G = std::make_shared<NegativeBilinearForm<TypeG>>(G_);

  // Create the block bilinear form.
  block_bf_ = std::make_shared<TypeBlockBF>(A_, B_, BT_, minus_G);

  // Create the Schur bilinear form.
  S_ = std::make_shared<TypeS>(P_Y_, B_, BT_, G_);

  // Create preconditioner for X_delta.
  space::OperatorOptions space_opts;
  space_opts.alpha_ = opts.alpha_;
  P_X_ = std::make_shared<TypePrecondX>(vec_X_out_.get(), vec_X_in_.get(),
                                        /* use_cache */ true, space_opts);
}

HeatEquation::HeatEquation(std::shared_ptr<TypeXVector> vec_X_in,
                           std::shared_ptr<TypeXVector> vec_X_out,
                           std::shared_ptr<TypeYVector> vec_Y_in,
                           std::shared_ptr<TypeYVector> vec_Y_out,
                           const HeatEquationOptions &opts)
    : HeatEquation(
          vec_X_in, vec_X_out, vec_Y_in, vec_Y_out,
          std::make_shared<TypeA>(vec_Y_in.get(), vec_Y_out.get()),
          std::make_shared<TypePrecondY>(vec_Y_out.get(), vec_Y_in.get()),
          opts) {}

HeatEquation::HeatEquation(const TypeXDelta &X_delta, const TypeYDelta &Y_delta,
                           const HeatEquationOptions &opts)
    : HeatEquation(std::make_shared<TypeXVector>(
                       X_delta.template DeepCopy<TypeXVector>()),
                   std::make_shared<TypeXVector>(
                       X_delta.template DeepCopy<TypeXVector>()),
                   std::make_shared<TypeYVector>(
                       Y_delta.template DeepCopy<TypeYVector>()),
                   std::make_shared<TypeYVector>(
                       Y_delta.template DeepCopy<TypeYVector>()),
                   opts) {}

HeatEquation::HeatEquation(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta,
    const HeatEquationOptions &opts)
    : HeatEquation(X_delta, GenerateYDelta(X_delta), opts) {}

}  // namespace applications
