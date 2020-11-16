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
HeatEquation::HeatEquation(std::shared_ptr<TypeXVector> vec_X,
                           std::shared_ptr<TypeYVector> vec_Y,
                           std::shared_ptr<TypeA> A,
                           std::shared_ptr<TypePrecondY> P_Y,
                           bool Yd_is_GenerateYDelta_Xd,
                           const HeatEquationOptions &opts)
    : vec_X_(vec_X), vec_Y_(vec_Y), A_(A), P_Y_(P_Y), opts_(opts) {
  space::OperatorOptions space_opts({.build_mat = opts_.build_space_mats});
  if (!A) A_ = std::make_shared<TypeA>(vec_Y.get(), opts.use_cache, space_opts);

  // Create first part of B.
  std::shared_ptr<TypeB_t> B_t;
  if (Yd_is_GenerateYDelta_Xd)
    // Create B_t making use of the fact that theta = vec_Ydd.
    B_t = std::make_shared<HeatEquation::TypeB_t>(
        vec_X_.get(), vec_Y_.get(), spacetime::GenerateSigma(*vec_X_, *vec_Y_),
        vec_Y_, opts_.use_cache, space_opts);
  else
    B_t = std::make_shared<TypeB_t>(vec_X_.get(), vec_Y_.get(), opts_.use_cache,
                                    space_opts);

  // Create B_s sharing sigma and theta.
  auto B_s =
      std::make_shared<TypeB_s>(vec_X_.get(), vec_Y_.get(), B_t->sigma(),
                                B_t->theta(), opts_.use_cache, space_opts);
  B_ = std::make_shared<TypeB>(B_t, B_s);

  // Create transpose of B sharing data with B.
  InitializeBT();

  // Create gamma_0'gamma_0 operator.
  G_ = std::make_shared<TypeG>(vec_X_.get(), opts_.use_cache, space_opts);

  // Create the negative gamma_0'gamma_0 operator.
  auto minus_G = std::make_shared<NegativeBilinearForm<TypeG>>(G_);

  // Initialize preconditioners (if necessary).
  if (!P_Y_) InitializePrecondY();
  InitializePrecondX();

  // Create the Schur bilinear form.
  S_ = std::make_shared<TypeS>(P_Y_, B_, BT_, G_);
}

HeatEquation::HeatEquation(std::shared_ptr<TypeXVector> vec_X,
                           std::shared_ptr<TypeYVector> vec_Y,
                           const HeatEquationOptions &opts)
    : HeatEquation(vec_X, vec_Y, nullptr, nullptr, false, opts) {}

HeatEquation::HeatEquation(const TypeXDelta &X_delta, const TypeYDelta &Y_delta,
                           const HeatEquationOptions &opts)
    : HeatEquation(std::make_shared<TypeXVector>(
                       X_delta.template DeepCopy<TypeXVector>()),
                   std::make_shared<TypeYVector>(
                       Y_delta.template DeepCopy<TypeYVector>()),
                   nullptr, nullptr, true, opts) {}

HeatEquation::HeatEquation(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta,
    const HeatEquationOptions &opts)
    : HeatEquation(X_delta, GenerateYDelta<DoubleTreeView>(X_delta), opts) {}

void HeatEquation::InitializeBT() {
  if (opts_.use_cache) {
    BT_ = B_->Transpose();
  } else {
    space::OperatorOptions space_opts({.build_mat = opts_.build_space_mats});
    auto BT_t = std::make_shared<
        BilinearForm<Time::TransportOperator, space::MassOperator,
                     OrthonormalWaveletFn, ThreePointWaveletFn>>(
        vec_Y_.get(), vec_X_.get(), B_->theta(), B_->sigma(), opts_.use_cache,
        space_opts);
    auto BT_s = std::make_shared<
        BilinearForm<Time::MassOperator, space::StiffnessOperator,
                     OrthonormalWaveletFn, ThreePointWaveletFn>>(
        vec_Y_.get(), vec_X_.get(), B_->theta(), B_->sigma(), opts_.use_cache,
        space_opts);
    BT_ = std::make_shared<SumBilinearForm<
        BilinearForm<Time::TransportOperator, space::MassOperator,
                     OrthonormalWaveletFn, ThreePointWaveletFn>,
        BilinearForm<Time::MassOperator, space::StiffnessOperator,
                     OrthonormalWaveletFn, ThreePointWaveletFn>>>(BT_t, BT_s);
  }
}

void HeatEquation::InitializePrecondX() {
  space::OperatorOptions space_opts({
      .build_mat = opts_.PXY_mg_build,
      .alpha = opts_.PX_alpha,
      .mg_cycles = opts_.PX_mg_cycles,
  });
  switch (opts_.PX_inv) {
    case HeatEquationOptions::SpaceInverse::DirectInverse:
      P_X_ = std::make_shared<spacetime::BlockDiagonalBilinearForm<
          space::XPreconditionerOperator<space::DirectInverse>,
          ThreePointWaveletFn, ThreePointWaveletFn>>(
          vec_X_.get(), vec_X_.get(), opts_.use_cache, space_opts);
      break;
    case HeatEquationOptions::SpaceInverse::Multigrid:
      P_X_ = std::make_shared<spacetime::BlockDiagonalBilinearForm<
          space::XPreconditionerOperator<space::MultigridPreconditioner>,
          ThreePointWaveletFn, ThreePointWaveletFn>>(
          vec_X_.get(), vec_X_.get(), opts_.use_cache, space_opts);
      break;
    default:
      assert(false);
  }
}

void HeatEquation::InitializePrecondY() {
  space::OperatorOptions space_opts({
      .build_mat = opts_.PXY_mg_build,
      .mg_cycles = opts_.PY_mg_cycles,
  });
  switch (opts_.PY_inv) {
    case HeatEquationOptions::SpaceInverse::DirectInverse:
      P_Y_ = std::make_shared<spacetime::BlockDiagonalBilinearForm<
          space::DirectInverse<space::StiffnessOperator>, OrthonormalWaveletFn,
          OrthonormalWaveletFn>>(vec_Y_.get(), vec_Y_.get(), opts_.use_cache,
                                 space_opts);
      break;
    case HeatEquationOptions::SpaceInverse::Multigrid:
      P_Y_ = std::make_shared<spacetime::BlockDiagonalBilinearForm<
          space::MultigridPreconditioner<space::StiffnessOperator>,
          OrthonormalWaveletFn, OrthonormalWaveletFn>>(
          vec_Y_.get(), vec_Y_.get(), opts_.use_cache, space_opts);
      break;
    default:
      assert(false);
  }
}

}  // namespace applications
