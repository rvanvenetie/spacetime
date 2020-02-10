#pragma once
#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../datastructures/multi_tree_vector.hpp"
#include "basis.hpp"
#include "sparse_vector.hpp"

namespace Time {

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
class BilinearForm {
 public:
  using WaveletBasisIn =
      std::remove_pointer_t<std::tuple_element_t<0, typename I_in::TupleNodes>>;
  using WaveletBasisOut = std::remove_pointer_t<
      std::tuple_element_t<0, typename I_out::TupleNodes>>;
  using ScalingBasisIn = typename FunctionTrait<WaveletBasisIn>::Scaling;
  using ScalingBasisOut = typename FunctionTrait<WaveletBasisOut>::Scaling;

  // Create a stateful BilinearForm.
  BilinearForm(I_in *root_vec_in, I_out *root_vec_out);

  void Apply() {
    InitializeInput();
    auto [_, f] = ApplyRecur(0, {}, {});

    // Copy data to the output tree.
    f.StoreInTree();
    vec_out_->ReadFromTree();
    f.RemoveFromTree();
  }

  void ApplyUpp() {
    InitializeInput();
    auto [_, f] = ApplyUppRecur(0, {}, {});

    // Copy data to the output tree.
    f.StoreInTree();
    vec_out_->ReadFromTree();
    f.RemoveFromTree();
  }

  void ApplyLow() {
    InitializeInput();
    auto f = ApplyLowRecur(0, {});

    // Copy data to the output tree.
    f.StoreInTree();
    vec_out_->ReadFromTree();
    f.RemoveFromTree();
  }

  // Debug function, O(n^2).
  Eigen::MatrixXd ToMatrix();

 protected:
  // Roots of the treeviews we are considering.
  I_in *vec_in_;
  I_out *vec_out_;

  // A flattened, levelwise view of the trees.
  std::vector<SparseVector<WaveletBasisIn>> lvl_vec_in_;
  std::vector<SparseIndices<WaveletBasisOut>> lvl_ind_out_;

  // Helper variables.
  SparseVector<WaveletBasisIn> empty_vec_in_;
  SparseIndices<WaveletBasisOut> empty_ind_out_;

  // Helper function to set the levelwise input vector.
  void InitializeInput();

  // Recursive apply.
  std::pair<SparseVector<ScalingBasisOut>, SparseVector<WaveletBasisOut>>
  ApplyRecur(size_t l, const SparseIndices<ScalingBasisOut> &Pi_out,
             const SparseVector<ScalingBasisIn> &d);

  // Recursive apply upper part.
  std::pair<SparseVector<ScalingBasisOut>, SparseVector<WaveletBasisOut>>
  ApplyUppRecur(size_t l, const SparseIndices<ScalingBasisOut> &Pi_out,
                const SparseVector<ScalingBasisIn> &d);

  // Recursive apply lower part.
  SparseVector<WaveletBasisOut> ApplyLowRecur(
      size_t l, const SparseVector<ScalingBasisIn> &d);

  // Index sets.
  std::pair<SparseIndices<ScalingBasisOut>, SparseIndices<ScalingBasisOut>>
  ConstructPiOut(const SparseIndices<ScalingBasisOut> &Pi_out);

  std::pair<SparseIndices<ScalingBasisIn>, SparseIndices<ScalingBasisIn>>
  ConstructPiIn(const SparseIndices<ScalingBasisIn> &Pi_in,
                const SparseIndices<ScalingBasisOut> &Pi_B_out);
};

// Helper functions .
template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
BilinearForm<Operator, I_in, I_out> CreateBilinearForm(I_in *root_vec_in,
                                                       I_out *root_vec_out) {
  return BilinearForm<Operator, I_in, I_out>(root_vec_in, root_vec_out);
}

// Helper functions .
template <template <typename, typename> class Operator, typename WaveletBasisIn,
          typename WaveletBasisOut>
BilinearForm<Operator, datastructures::NodeVector<WaveletBasisIn>,
             datastructures::NodeVector<WaveletBasisOut>>
CreateBilinearForm(const datastructures::TreeVector<WaveletBasisIn> &vec_in,
                   const datastructures::TreeVector<WaveletBasisOut> &vec_out) {
  return {vec_in.root(), vec_out.root()};
}

}  // namespace Time

#include "bilinear_form.ipp"
