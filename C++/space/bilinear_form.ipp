#pragma once
#include "bilinear_form.hpp"

namespace space {
template <typename Operator, typename I_in, typename I_out>
BilinearForm<Operator, I_in, I_out>::BilinearForm(
    std::shared_ptr<I_in> root_vec_in, std::shared_ptr<I_out> root_vec_out,
    bool dirichlet_boundary)
    : vec_in_(root_vec_in), vec_out_(root_vec_out) {
  assert(root_vec_in->is_root());
  assert(root_vec_out->is_root());
  auto vec_in_nodes = vec_in_->Bfs();
  auto vec_out_nodes = vec_out_->Bfs();
  assert(vec_in_nodes.size());
  assert(vec_out_nodes.size());

  // Determine whether the input and output vector coincide.
  symmetric_ = (vec_in_nodes.size() == vec_out_nodes.size());
  if (symmetric_)
    for (size_t i = 0; i < vec_in_nodes.size(); ++i)
      if (vec_in_nodes[i]->node() != vec_out_nodes[i]->node()) {
        symmetric_ = false;
        break;
      }

  // If this applicator is symmetric, there is not a lot to do.
  if (symmetric_) {
    triang_ = std::make_unique<TriangulationView>(vec_in_);
  } else {
    // This operator is not symmetric, calculate a union.
    vec_union_ =
        std::make_shared<datastructures::NodeVector<HierarchicalBasisFn>>(
            vec_in_->node());
    vec_union_->Union(vec_in_);
    vec_union_->Union(vec_out_);
    triang_ = std::make_unique<TriangulationView>(vec_union_);
  }
  operator_ = std::make_unique<Operator>(*triang_, dirichlet_boundary);
}

template <typename Operator, typename I_in, typename I_out>
void BilinearForm<Operator, I_in, I_out>::Apply() {
  if (symmetric_)
    // Apply the operator in SS.
    return vec_out_->FromVector(operator_->Apply(vec_in_->ToVector()));

  // Not symmetric, we must do some magic tricks.
  auto lambda_copy = [](const auto &new_node, const auto &old_node) {
    new_node->set_value(old_node->value());
  };
  vec_union_->Reset();
  vec_union_->Union(vec_in_, datastructures::func_false, lambda_copy);
  // Apply the operator in SS.
  vec_union_->FromVector(operator_->Apply(vec_union_->ToVector()));
  vec_out_->Union(vec_union_, datastructures::func_false, lambda_copy);
}

template <typename Operator, typename I_in, typename I_out>
Eigen::MatrixXd BilinearForm<Operator, I_in, I_out>::ToMatrix() {
  auto nodes = vec_in_->Bfs();
  Eigen::MatrixXd A =
      Eigen::MatrixXd::Zero(vec_out_->Bfs().size(), nodes.size());
  for (int i = 0; i < nodes.size(); ++i) {
    for (int j = 0; j < nodes.size(); ++j) {
      nodes[j]->set_value(0);
    }
    nodes[i]->set_value(1);
    Apply();
    auto nodes_out = vec_out_->Bfs();
    for (int j = 0; j < nodes_out.size(); ++j) {
      A(j, i) = nodes_out[j]->value();
    }
  }
  return A;
}

}  // namespace space
