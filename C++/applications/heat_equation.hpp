#pragma once
#include "../datastructures/double_tree_view.hpp"
#include "../spacetime/bilinear_form.hpp"

namespace applications {
using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using spacetime::BilinearForm;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

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
  using TypeBlockMat =
      BlockBilinearForm<TypeA, TypeB, TypeBT, NegativeBilinearForm<TypeG>>;

  HeatEquation(
      const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta,
      const DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn> &Y_delta);
  HeatEquation(
      const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta);

  // This returns *shared_ptrs* to the respective operators.
  auto A() { return A_; }
  auto B() { return B_; };
  auto BT() { return BT_; }
  auto G() { return G_; }
  auto BlockMat() { return block_mat_; }

  auto vec_X_in() { return &vec_X_in_; }
  auto vec_X_out() { return &vec_X_out_; }
  auto vec_Y_in() { return &vec_Y_in_; }
  auto vec_Y_out() { return &vec_Y_out_; }

 protected:
  std::shared_ptr<TypeA> A_;
  std::shared_ptr<TypeB> B_;
  std::shared_ptr<TypeBT> BT_;
  std::shared_ptr<TypeG> G_;
  std::shared_ptr<TypeBlockMat> block_mat_;

  // Store doubletree vectors for X_delta input and output.
  DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> vec_X_in_,
      vec_X_out_;

  // Store doubletree vectors for Y_delta input and output.
  DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn> vec_Y_in_,
      vec_Y_out_;
};

}  // namespace applications
