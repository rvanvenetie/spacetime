#pragma once
#include <utility>

#include "operators.hpp"
#include "triangulation_view_new.hpp"

namespace space {

template <typename Operator, typename I_in, typename I_out = I_in>
class BilinearForm {
 public:
  BilinearForm(I_in* root_vec_in, I_out* root_vec_out,
               const OperatorOptions& opts);

  void Apply();

  auto Transpose() const {
    auto transpose = BilinearForm<Operator, I_out, I_in>();
    transpose.vec_in_ = vec_out_;
    transpose.vec_out_ = vec_in_;
    transpose.symmetric_ = symmetric_;
    transpose.triang_ = triang_;
    transpose.operator_ = operator_;
    if (symmetric_) {
      transpose.nodes_vec_in_ = nodes_vec_out_;
      transpose.nodes_vec_out_ = nodes_vec_in_;
    } else {
      transpose.vec_union_ = vec_union_;
      transpose.nodes_vec_union_ = nodes_vec_union_;
    }
    return transpose;
  }

  Eigen::MatrixXd ToMatrix();

 protected:
  // Protected constructor, and give transpose operator access.
  BilinearForm() : vec_in_(nullptr), vec_out_(nullptr) {}
  friend BilinearForm<Operator, I_out, I_in>;

  I_in* vec_in_;
  I_out* vec_out_;
  bool symmetric_;
  std::shared_ptr<TriangulationViewNew> triang_;
  std::shared_ptr<Operator> operator_;

  // If symmetric, a flattened bfs view of input/output vectors.
  std::shared_ptr<std::vector<I_in*>> nodes_vec_in_;
  std::shared_ptr<std::vector<I_out*>> nodes_vec_out_;

  // If not symmetric, a treevector and flattened list of the union.
  std::shared_ptr<datastructures::TreeVector<HierarchicalBasisFn>> vec_union_;
  std::shared_ptr<std::vector<datastructures::NodeVector<HierarchicalBasisFn>*>>
      nodes_vec_union_;
};

// Helper functions.
template <typename Operator, typename I_in, typename I_out>
BilinearForm<Operator, I_in, I_out> CreateBilinearForm(
    I_in* root_vec_in, I_out* root_vec_out,
    const OperatorOptions& opts = OperatorOptions()) {
  return BilinearForm<Operator, I_in, I_out>(root_vec_in, root_vec_out, opts);
}

template <typename Operator>
auto CreateBilinearForm(
    const datastructures::TreeVector<HierarchicalBasisFn>& vec_in,
    const datastructures::TreeVector<HierarchicalBasisFn>& vec_out,
    const OperatorOptions& opts = OperatorOptions()) {
  return BilinearForm<Operator,
                      datastructures::NodeVector<HierarchicalBasisFn>>(
      vec_in.root(), vec_out.root(), opts);
}

}  // namespace space

#include "bilinear_form.ipp"
