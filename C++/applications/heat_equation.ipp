#pragma once
#include "heat_equation.hpp"

namespace applications {
using spacetime::GenerateYDelta;

template <bool use_cache>
HeatEquation<use_cache>::HeatEquation(std::shared_ptr<TypeXVector> vec_X_in,
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
  block_ = std::make_shared<TypeBlockMat>(A_, B_, BT_, minus_G);
  // Create the Schur matrix.
  schur_ = std::make_shared<TypeSchurMat>(A_inv_, B_, BT_, G_);
  precond_X_ =
      std::make_shared<TypePrecondX>(vec_X_in_.get(), vec_X_out_.get());
}

template <bool use_cache>
HeatEquation<use_cache>::HeatEquation(std::shared_ptr<TypeXVector> vec_X_in,
                                      std::shared_ptr<TypeXVector> vec_X_out,
                                      std::shared_ptr<TypeYVector> vec_Y_in,
                                      std::shared_ptr<TypeYVector> vec_Y_out)
    : HeatEquation<use_cache>(
          vec_X_in, vec_X_out, vec_Y_in, vec_Y_out,
          std::make_shared<TypeA>(vec_Y_in.get(), vec_Y_out.get(), use_cache),
          std::make_shared<TypeAinv>(vec_Y_out.get(), vec_Y_in.get(),
                                     use_cache)) {}

template <bool use_cache>
HeatEquation<use_cache>::HeatEquation(const TypeXDelta &X_delta,
                                      const TypeYDelta &Y_delta)
    : HeatEquation<use_cache>(std::make_shared<TypeXVector>(
                                  X_delta.template DeepCopy<TypeXVector>()),
                              std::make_shared<TypeXVector>(
                                  X_delta.template DeepCopy<TypeXVector>()),
                              std::make_shared<TypeYVector>(
                                  Y_delta.template DeepCopy<TypeYVector>()),
                              std::make_shared<TypeYVector>(
                                  Y_delta.template DeepCopy<TypeYVector>())) {}

template <bool use_cache>
HeatEquation<use_cache>::HeatEquation(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta)
    : HeatEquation<use_cache>(X_delta, GenerateYDelta(X_delta)) {}

}  // namespace applications
