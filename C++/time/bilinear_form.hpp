#pragma once
#include "datastructures/include.hpp"

#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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
    FinalizeOutput(f);
  }

  void ApplyUpp() {
    InitializeInput();
    auto [_, f] = ApplyUppRecur(0, {}, {});
    FinalizeOutput(f);
  }

  void ApplyLow() {
    InitializeInput();
    auto f = ApplyLowRecur(0, {});
    FinalizeOutput(f);
  }

  auto Transpose() const {
    auto transpose = BilinearForm<Operator, I_out, I_in>();
    transpose.vec_in_ = vec_out_;
    transpose.vec_out_ = vec_in_;
    transpose.nodes_vec_in_ = nodes_vec_out_;
    transpose.nodes_vec_out_ = nodes_vec_in_;
    transpose.InitializeOutput();
    return transpose;
  }

  // Debug function, O(n^2).
  Eigen::MatrixXd ToMatrix();

 protected:
  // Protected constructor, and give transpose operator access.
  BilinearForm() : vec_in_(nullptr), vec_out_(nullptr) {}
  friend BilinearForm<Operator, I_out, I_in>;

  // Roots of the treeviews we are considering.
  I_in *vec_in_;
  I_out *vec_out_;

  // A flattened (levelwise) view of input/output vectors.
  std::shared_ptr<std::vector<std::vector<I_in *>>> nodes_vec_in_;
  std::shared_ptr<std::vector<std::vector<I_out *>>> nodes_vec_out_;

  // Another flattened (levelwise) view of the vectors, in another data format.
  std::vector<SparseVector<WaveletBasisIn>> lvl_vec_in_;
  std::vector<SparseIndices<WaveletBasisOut>> lvl_ind_out_;

  // Helper variables.
  SparseVector<WaveletBasisIn> empty_vec_in_;
  SparseIndices<WaveletBasisOut> empty_ind_out_;

  // Helper function to set the levelwise input/output vector.
  void InitializeOutput();
  void InitializeInput();
  void FinalizeOutput(const SparseVector<WaveletBasisOut> &f);

  // Recursive apply.
  std::pair<SparseVector<ScalingBasisOut>, SparseVector<WaveletBasisOut>>
  ApplyRecur(size_t l, SparseIndices<ScalingBasisOut> &&Pi_out,
             const SparseVector<ScalingBasisIn> &d);

  // Recursive apply upper part.
  std::pair<SparseVector<ScalingBasisOut>, SparseVector<WaveletBasisOut>>
  ApplyUppRecur(size_t l, SparseIndices<ScalingBasisOut> &&Pi_out,
                const SparseVector<ScalingBasisIn> &d);

  // Recursive apply lower part.
  SparseVector<WaveletBasisOut> ApplyLowRecur(
      size_t l, const SparseVector<ScalingBasisIn> &d);

  // Index sets.
  std::pair<SparseIndices<ScalingBasisOut>, SparseIndices<ScalingBasisOut>>
  ConstructPiOut(SparseIndices<ScalingBasisOut> &&Pi_out,
                 bool construct_Pi_A_out = true);

  SparseIndices<ScalingBasisIn> ConstructPiBIn(
      SparseIndices<ScalingBasisIn> &&Pi_in,
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
