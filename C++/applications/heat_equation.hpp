#pragma once
#include "../datastructures/double_tree_view.hpp"
#include "../spacetime/bilinear_form.hpp"

namespace applications {

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using spacetime::BilinearForm;
using spacetime::BlockBilinearForm;
using spacetime::BlockDiagonalBilinearForm;
using spacetime::NegativeBilinearForm;
using spacetime::RemapBilinearForm;
using spacetime::SchurBilinearForm;
using spacetime::SumBilinearForm;
using spacetime::TransposeBilinearForm;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

struct HeatEquationOptions {
  // The alpha used in the preconditioner on X.
  double alpha_ = 0.3;
};

// Base class for constructing the operators necessary.
class HeatEquation {
 public:
  // The symmetric operator acting from Y_delta to Y_delta.
  using TypeA = BilinearForm<Time::MassOperator, space::StiffnessOperator,
                             OrthonormalWaveletFn, OrthonormalWaveletFn>;
  // The (transport) part of operator B acting from X_delta to Y_delta.
  using TypeB_t = BilinearForm<Time::TransportOperator, space::MassOperator,
                               ThreePointWaveletFn, OrthonormalWaveletFn>;
  // The (mass_stiff) part of operator B acting from X_delta to Y_delta.
  using TypeB_s = BilinearForm<Time::MassOperator, space::StiffnessOperator,
                               ThreePointWaveletFn, OrthonormalWaveletFn>;

  // The operator B is the sum of these two operators.
  using TypeB = SumBilinearForm<TypeB_t, TypeB_s>;

  // The transpose of B is the sum of the transpose of these two operators.
  // With the output/input vectors correctly remapped.
  using TypeBT =
      RemapBilinearForm<SumBilinearForm<TransposeBilinearForm<TypeB_t>,
                                        TransposeBilinearForm<TypeB_s>>>;

  // The trace operator maps between X_delta and X_delta.
  using TypeG = BilinearForm<Time::ZeroEvalOperator, space::MassOperator,
                             ThreePointWaveletFn, ThreePointWaveletFn>;

  // The block matrix necessary for the solving using minres.
  using TypeBlockBF =
      BlockBilinearForm<TypeA, TypeB, TypeBT, NegativeBilinearForm<TypeG>>;

  // Preconditioners.
  using TypePrecondY = spacetime::BlockDiagonalBilinearForm<
      space::DirectInverse<space::StiffnessOperator>, OrthonormalWaveletFn,
      OrthonormalWaveletFn>;
  using TypePrecondX = spacetime::BlockDiagonalBilinearForm<
      space::XPreconditionerOperator<space::DirectInverse>, ThreePointWaveletFn,
      ThreePointWaveletFn>;

  // Schur complement.
  using TypeS = SchurBilinearForm<TypePrecondY, TypeB, TypeBT, TypeG>;

  using TypeXDelta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeYDelta = DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>;
  using TypeXVector =
      DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>;
  using TypeYVector =
      DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>;

  HeatEquation(std::shared_ptr<TypeXVector> vec_X_in,
               std::shared_ptr<TypeXVector> vec_X_out,
               std::shared_ptr<TypeYVector> vec_Y_in,
               std::shared_ptr<TypeYVector> vec_Y_out, std::shared_ptr<TypeA> A,
               std::shared_ptr<TypePrecondY> P_Y,
               const HeatEquationOptions &opts = HeatEquationOptions());
  HeatEquation(std::shared_ptr<TypeXVector> vec_X_in,
               std::shared_ptr<TypeXVector> vec_X_out,
               std::shared_ptr<TypeYVector> vec_Y_in,
               std::shared_ptr<TypeYVector> vec_Y_out,
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
  auto BlockBF() { return block_bf_; }

  auto P_Y() { return P_Y_; }
  auto P_X() { return P_X_; }

  auto S() { return S_; }

  auto vec_X_in() { return vec_X_in_.get(); }
  auto vec_X_out() { return vec_X_out_.get(); }
  auto vec_Y_in() { return vec_Y_in_.get(); }
  auto vec_Y_out() { return vec_Y_out_.get(); }

 protected:
  HeatEquationOptions opts_;

  // Operators.
  std::shared_ptr<TypeA> A_;
  std::shared_ptr<TypeB> B_;
  std::shared_ptr<TypeBT> BT_;
  std::shared_ptr<TypeG> G_;
  std::shared_ptr<TypeBlockBF> block_bf_;

  // Preconditioners.
  std::shared_ptr<TypePrecondY> P_Y_;
  std::shared_ptr<TypePrecondX> P_X_;

  // Schur complement.
  std::shared_ptr<TypeS> S_;

  // Store doubletree vectors for X_delta input and output.
  std::shared_ptr<TypeXVector> vec_X_in_, vec_X_out_;

  // Store doubletree vectors for Y_delta input and output.
  std::shared_ptr<TypeYVector> vec_Y_in_, vec_Y_out_;
};

}  // namespace applications
