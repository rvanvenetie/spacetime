#pragma once
#include "../datastructures/double_tree_view.hpp"
#include "../spacetime/bilinear_form.hpp"

namespace applications {
using datastructures::BlockBilinearForm;
using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using datastructures::NegativeBilinearForm;
using datastructures::RemapBilinearForm;
using datastructures::SchurBilinearForm;
using datastructures::SumBilinearForm;
using datastructures::TransposeBilinearForm;
using space::HierarchicalBasisFn;
using spacetime::BilinearForm;
using spacetime::BlockDiagonalBilinearForm;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

// Base class for constructing the operators necessary.
template <bool UseCache = true>
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
  using TypeBT = typename std::conditional<
      UseCache,
      RemapBilinearForm<SumBilinearForm<TransposeBilinearForm<TypeB_t>,
                                        TransposeBilinearForm<TypeB_s>>>,
      SumBilinearForm<
          BilinearForm<Time::TransportOperator, space::MassOperator,
                       OrthonormalWaveletFn, ThreePointWaveletFn>,
          BilinearForm<Time::MassOperator, space::StiffnessOperator,
                       OrthonormalWaveletFn, ThreePointWaveletFn>>>::type;

  // The trace operator maps between X_delta and X_delta.
  using TypeG = BilinearForm<Time::ZeroEvalOperator, space::MassOperator,
                             ThreePointWaveletFn, ThreePointWaveletFn>;

  // The block matrix necessary for the solving using minres.
  using TypeBlockBF =
      BlockBilinearForm<TypeA, TypeB, TypeBT, NegativeBilinearForm<TypeG>>;

  // Types necessary for the Schur complement matrix.
  using TypeAinv = spacetime::BlockDiagonalBilinearForm<
      space::DirectInverse<space::StiffnessOperator>, OrthonormalWaveletFn,
      OrthonormalWaveletFn>;
  using TypeSchurBF = SchurBilinearForm<TypeAinv, TypeB, TypeBT, TypeG>;
  using TypePrecondX = spacetime::BlockDiagonalBilinearForm<
      space::XPreconditionerOperator<space::DirectInverse>, ThreePointWaveletFn,
      ThreePointWaveletFn>;

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
               std::shared_ptr<TypeAinv> Ainv);
  HeatEquation(std::shared_ptr<TypeXVector> vec_X_in,
               std::shared_ptr<TypeXVector> vec_X_out,
               std::shared_ptr<TypeYVector> vec_Y_in,
               std::shared_ptr<TypeYVector> vec_Y_out);
  HeatEquation(const TypeXDelta &X_delta, const TypeYDelta &Y_delta);
  HeatEquation(const TypeXDelta &X_delta);

  // This returns *shared_ptrs* to the respective operators.
  auto A() { return A_; }
  auto B() { return B_; };
  auto BT() { return BT_; }
  auto G() { return G_; }
  auto BlockBF() { return block_bf_; }

  auto Ainv() { return A_inv_; }
  auto SchurBF() { return schur_bf_; }
  auto PrecondX() { return precond_X_; }

  auto vec_X_in() { return vec_X_in_.get(); }
  auto vec_X_out() { return vec_X_out_.get(); }
  auto vec_Y_in() { return vec_Y_in_.get(); }
  auto vec_Y_out() { return vec_Y_out_.get(); }

 protected:
  void InitializeBT();

  std::shared_ptr<TypeA> A_;
  std::shared_ptr<TypeB> B_;
  std::shared_ptr<TypeBT> BT_;
  std::shared_ptr<TypeG> G_;
  std::shared_ptr<TypeBlockBF> block_bf_;

  // Schur complement stuff.
  std::shared_ptr<TypeAinv> A_inv_;
  std::shared_ptr<TypeSchurBF> schur_bf_;
  std::shared_ptr<TypePrecondX> precond_X_;

  // Store doubletree vectors for X_delta input and output.
  std::shared_ptr<TypeXVector> vec_X_in_, vec_X_out_;

  // Store doubletree vectors for Y_delta input and output.
  std::shared_ptr<TypeYVector> vec_Y_in_, vec_Y_out_;
};

template <>
void HeatEquation<false>::InitializeBT();
template <>
void HeatEquation<true>::InitializeBT();

extern template class HeatEquation<false>;
extern template class HeatEquation<true>;

}  // namespace applications
