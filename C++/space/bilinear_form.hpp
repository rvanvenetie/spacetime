#pragma once
#include <chrono>
#include <utility>

#include "operators.hpp"
#include "triangulation_view.hpp"

namespace space {

template <typename Operator, typename I_in, typename I_out = I_in>
class BilinearForm {
 public:
  std::chrono::duration<double> time_bfs_{0};
  std::chrono::duration<double> time_tview_{0};
  std::chrono::duration<double> time_operator_{0};

  BilinearForm(I_in* root_vec_in, I_out* root_vec_out,
               const OperatorOptions& opts);

  void Apply();
  std::shared_ptr<TriangulationView> TView() { return triang_; }

  auto Transpose() const {
    auto transpose = BilinearForm<Operator, I_out, I_in>();
    if (inclusion_type_ == InclusionType::Subset)
      transpose.inclusion_type_ = InclusionType::Superset;
    else if (inclusion_type_ == InclusionType::Equal)
      transpose.inclusion_type_ = InclusionType::Equal;
    else if (inclusion_type_ == InclusionType::Superset)
      transpose.inclusion_type_ = InclusionType::Subset;
    else
      assert(false);

    transpose.vec_in_ = vec_out_;
    transpose.vec_out_ = vec_in_;
    transpose.triang_ = triang_;
    transpose.operator_ = operator_;
    transpose.nodes_vec_in_ = nodes_vec_out_;
    transpose.nodes_vec_out_ = nodes_vec_in_;
    return transpose;
  }

  double TimeCreate() const { return time_create_.count(); }
  double TimePerApply() const {
    if (num_apply_ == 0)
      return 0;
    else
      return time_apply_.count() / num_apply_;
  }

  Eigen::MatrixXd ToMatrix();

  // These are the functions that must be implemented for Eigen to work.
  Eigen::Index rows() const { return nodes_vec_out_->size(); }
  Eigen::Index cols() const { return nodes_vec_in_->size(); }

  template <typename Rhs>
  Eigen::VectorXd operator*(const Eigen::MatrixBase<Rhs>& x) const {
    FromVector(*nodes_vec_in_, x);
    const_cast<BilinearForm*>(this)->Apply();
    return ToVector(*nodes_vec_out_);
  }

 protected:
  // Protected constructor, and give transpose operator access.
  BilinearForm() : vec_in_(nullptr), vec_out_(nullptr) {}
  friend BilinearForm<Operator, I_out, I_in>;

  // What kind of ordering do we have?
  // vec_in < vec_out, vec_in == vec_out, or vec_in > vec_out.
  enum InclusionType { Subset, Equal, Superset };
  InclusionType inclusion_type_;

  I_in* vec_in_;
  I_out* vec_out_;
  std::shared_ptr<TriangulationView> triang_;
  std::shared_ptr<Operator> operator_;

  // A flattened bfs view of input/output vectors.
  std::shared_ptr<std::vector<I_in*>> nodes_vec_in_;
  std::shared_ptr<std::vector<I_out*>> nodes_vec_out_;

  // Some timing results -- debug info.
  std::chrono::duration<double> time_create_;
  std::chrono::duration<double> time_apply_{0};
  size_t num_apply_ = 0;
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
