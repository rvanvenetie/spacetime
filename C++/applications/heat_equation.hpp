#pragma once
// clang-format off
#include "spacetime/includes.hpp"
#include "spacetime/bilinear_form.hpp"
// clang-format on

namespace applications {

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using spacetime::BilinearForm;
using spacetime::BilinearFormBase;
using spacetime::BlockDiagonalBilinearForm;
using spacetime::NegativeBilinearForm;
using spacetime::SchurBilinearForm;
using spacetime::SumBilinearForm;
using spacetime::SymmetricBilinearForm;
using spacetime::TransposeBilinearForm;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

struct HeatEquationOptions {
  // Whether or not to cache the bilinear forms.
  bool use_cache_ = true;

  // The alpha used in the preconditioner on X.
  double P_X_alpha_ = 0.3;

  // Options for the inversion of space matrices, necessary for preconditioners.
  enum SpaceInverse { DirectInverse, Multigrid };
  SpaceInverse P_X_inv_ = SpaceInverse::DirectInverse;
  SpaceInverse P_Y_inv_ = SpaceInverse::DirectInverse;

  // If a multigrid Preconditioner is chosen, this sets the number of cycles.
  size_t P_X_mg_cycles_ = 5;
  size_t P_Y_mg_cycles_ = 5;
};

// Base class for constructing the operators necessary.
class HeatEquation {
 public:
  // DoubleTreeView/DoubleTreeVectors.
  using TypeXDelta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeYDelta = DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>;
  using TypeXVector =
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeYVector =
      DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>;

  // The symmetric operator acting from Y_delta to Y_delta.
  using TypeA =
      SymmetricBilinearForm<Time::MassOperator, space::StiffnessOperator,
                            OrthonormalWaveletFn>;
  // The (transport) part of operator B acting from X_delta to Y_delta.
  using TypeB_t = BilinearForm<Time::TransportOperator, space::MassOperator,
                               ThreePointWaveletFn, OrthonormalWaveletFn>;
  // The (mass_stiff) part of operator B acting from X_delta to Y_delta.
  using TypeB_s = BilinearForm<Time::MassOperator, space::StiffnessOperator,
                               ThreePointWaveletFn, OrthonormalWaveletFn>;

  // The operator B is the sum of these two operators.
  using TypeB = SumBilinearForm<TypeB_t, TypeB_s>;

  // The transpose of B is the sum of the transpose of these two operators.
  using TypeBT = BilinearFormBase<TypeYVector, TypeXVector>;

  // The trace operator maps between X_delta and X_delta.
  using TypeG = SymmetricBilinearForm<Time::ZeroEvalOperator,
                                      space::MassOperator, ThreePointWaveletFn>;

  // Preconditioners.
  using TypePrecondY = BilinearFormBase<TypeYVector, TypeYVector>;
  using TypePrecondX = BilinearFormBase<TypeXVector, TypeXVector>;

  // Schur complement.
  using TypeS = SchurBilinearForm<TypePrecondY, TypeB, TypeBT, TypeG>;

  HeatEquation(std::shared_ptr<TypeXVector> vec_X,
               std::shared_ptr<TypeYVector> vec_Y, std::shared_ptr<TypeA> A,
               std::shared_ptr<TypePrecondY> P_Y = nullptr,
               const HeatEquationOptions &opts = HeatEquationOptions());
  HeatEquation(std::shared_ptr<TypeXVector> vec_X,
               std::shared_ptr<TypeYVector> vec_Y,
               const HeatEquationOptions &opts = HeatEquationOptions());
  HeatEquation(const TypeXDelta &X_delta, const TypeYDelta &Y_delta,
               const HeatEquationOptions &opts = HeatEquationOptions());
  HeatEquation(const TypeXDelta &X_delta,
               const HeatEquationOptions &opts = HeatEquationOptions());

  // This returns *shared_ptrs* to the respective operators.
  auto A() { return A_; }
  auto B() { return B_; };
  auto BT() { return BT_; }
  auto G() { return G_; }

  auto P_Y() { return P_Y_; }
  auto P_X() { return P_X_; }

  auto S() { return S_; }

  auto vec_X() { return vec_X_.get(); }
  auto vec_Y() { return vec_Y_.get(); }

 protected:
  HeatEquationOptions opts_;

  // Operators.
  std::shared_ptr<TypeA> A_;
  std::shared_ptr<TypeB> B_;
  std::shared_ptr<TypeBT> BT_;
  std::shared_ptr<TypeG> G_;

  // Preconditioners.
  std::shared_ptr<TypePrecondY> P_Y_;
  std::shared_ptr<TypePrecondX> P_X_;

  // Schur complement.
  std::shared_ptr<TypeS> S_;

  // Store doubletree vectors for X_delta input and output.
  std::shared_ptr<TypeXVector> vec_X_;

  // Store doubletree vectors for Y_delta input and output.
  std::shared_ptr<TypeYVector> vec_Y_;

  void InitializeBT();
  // Constructors for preconditioners.
  void InitializePrecondX();
  void InitializePrecondY();
};

}  // namespace applications
