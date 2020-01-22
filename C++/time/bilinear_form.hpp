#pragma once
#include <Eigen/Dense>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../datastructures/multi_tree_vector.hpp"
#include "basis.hpp"
#include "sparse_vector.hpp"

namespace Time {

template <template <typename, typename> class Operator, typename WaveletBasisIn,
          typename WaveletBasisOut>
class BilinearForm {
 public:
  using ScalingBasisIn = typename FunctionTrait<WaveletBasisIn>::Scaling;
  using ScalingBasisOut = typename FunctionTrait<WaveletBasisOut>::Scaling;

  BilinearForm(datastructures::TreeVector<WaveletBasisOut> *vec_out)
      : vec_out_(vec_out) {
    // Slice the output vector into levelwise sparse indices.
    for (const auto &node : vec_out->Bfs()) {
      assert(node->level() >= 0 && node->level() <= lvl_ind_out_.size());
      if (node->level() == lvl_ind_out_.size()) lvl_ind_out_.emplace_back();
      lvl_ind_out_[node->level()].emplace_back(node->node().get());
    }
  }

  void Apply(const datastructures::TreeVector<WaveletBasisIn> &vec_in) {
    // Slice the input vector into levelwise sparse vectors.
    for (const auto &node : vec_in.Bfs()) {
      assert(node->level() >= 0 && node->level() <= lvl_vec_in_.size());
      if (node->level() == lvl_vec_in_.size()) lvl_vec_in_.emplace_back();
      lvl_vec_in_[node->level()].emplace_back(node->node().get(),
                                              node->value());
    }

    auto [_, f] = ApplyRecur(0, {}, {});

    // Copy data to the output tree.
    f.StoreInTree();
    for (auto nv : vec_out_->Bfs()) {
      nv->set_value(*nv->node()->template data<double>());
      nv->node()->reset_data();
    }
    lvl_vec_in_.clear();
  }

  // Debug function, O(n^2).
  Eigen::MatrixXd ToMatrix(
      const datastructures::TreeView<WaveletBasisIn> &tree_in) {
    auto indices_in = tree_in.Bfs();
    Eigen::MatrixXd A =
        Eigen::MatrixXd::Zero(vec_out_->Bfs().size(), indices_in.size());
    for (int i = 0; i < indices_in.size(); ++i) {
      auto vec_in =
          tree_in
              .template DeepCopy<datastructures::TreeVector<WaveletBasisIn>>();
      vec_in.Bfs()[i]->set_value(1);
      Apply(vec_in);
      auto nodes_out = vec_out_->Bfs();
      for (int j = 0; j < nodes_out.size(); ++j) {
        A(j, i) = nodes_out[j]->value();
      }
    }
    return A;
  }

  // Small helper function.
  datastructures::TreeVector<WaveletBasisOut> *vec_out() const {
    return vec_out_;
  }

 protected:
  std::vector<SparseVector<WaveletBasisIn>> lvl_vec_in_;
  std::vector<SparseIndices<WaveletBasisOut>> lvl_ind_out_;
  datastructures::TreeVector<WaveletBasisOut> *vec_out_;

  // Recursive apply.
  std::pair<SparseVector<ScalingBasisOut>, SparseVector<WaveletBasisOut>>
  ApplyRecur(size_t l, const SparseIndices<ScalingBasisOut> &Pi_out,
             const SparseVector<ScalingBasisIn> &d);

  // Index sets.
  std::pair<SparseIndices<ScalingBasisOut>, SparseIndices<ScalingBasisOut>>
  ConstructPiOut(const SparseIndices<ScalingBasisOut> &Pi_out);

  std::pair<SparseIndices<ScalingBasisIn>, SparseIndices<ScalingBasisIn>>
  ConstructPiIn(const SparseIndices<ScalingBasisIn> &Pi_in,
                const SparseIndices<ScalingBasisOut> &Pi_B_out);
};

}  // namespace Time

#include "bilinear_form.ipp"
