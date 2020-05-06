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
                           std::shared_ptr<TypeAY> AY,
                           std::shared_ptr<TypePrecondY> P_Y,
                           bool Yd_is_GenerateYDelta_Xd,
                           const HeatEquationOptions &opts)
    : vec_X_(vec_X), vec_Y_(vec_Y), AY_(AY), P_Y_(P_Y), opts_(opts) {
  {
    space::OperatorOptions space_opts({.build_mat = opts_.build_space_mats});
    if (!AY)
      AY_ = std::make_shared<TypeAY>(vec_Y_.get(), opts.use_cache, space_opts);
    if (Yd_is_GenerateYDelta_Xd)
      // Create C making use of the fact that theta = vec_Ydd.
      C_ = std::make_shared<TypeC>(vec_X_.get(), vec_Y_.get(),
                                   spacetime::GenerateSigma(*vec_X_, *vec_Y_),
                                   vec_Y_, opts_.use_cache, space_opts);
    else
      C_ = std::make_shared<TypeC>(vec_X_.get(), vec_Y_.get(), opts_.use_cache,
                                   space_opts);
    if (opts_.use_cache) {
      CT_ = C_->Transpose();
    } else {
      CT_ = std::make_shared<
          BilinearForm<Time::TransportOperator, space::MassOperator,
                       OrthonormalWaveletFn, ThreePointWaveletFn>>(
          vec_Y_.get(), vec_X_.get(), C_->theta(), C_->sigma(), opts_.use_cache,
          space_opts);
    }
    AX_ = std::make_shared<TypeAX>(vec_X_.get(), opts.use_cache, space_opts);
    G1_ = std::make_shared<TypeGT>(vec_X_.get(), opts_.use_cache, space_opts);
  }

  // Initialize preconditioners.
  if (!P_Y) {
    space::OperatorOptions space_opts({
        .build_mat = opts_.PXY_mg_build,
        .mg_cycles = opts_.PY_mg_cycles,
    });
    switch (opts_.PY_inv) {
      case HeatEquationOptions::SpaceInverse::DirectInverse:
        P_Y_ = std::make_shared<spacetime::BlockDiagonalBilinearForm<
            space::DirectInverse<space::StiffnessOperator>,
            OrthonormalWaveletFn, OrthonormalWaveletFn>>(
            vec_Y_.get(), vec_Y_.get(), opts_.use_cache, space_opts);
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
    {
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

    // Create the Schur bilinear form.
    S_ = std::make_shared<TypeS>(P_Y_, C_, CT_, AX_, G1_);
  }
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

}  // namespace applications
