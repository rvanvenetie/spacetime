#pragma once
#include "bilinear_form.hpp"

namespace space {
template <typename Operator, typename I_in, typename I_out>
BilinearForm<Operator, I_in, I_out>::BilinearForm(I_in* root_vec_in,
                                                  I_out* root_vec_out,
                                                  const OperatorOptions& opts)
    : vec_in_(root_vec_in), vec_out_(root_vec_out) {
  assert(root_vec_in->is_root());
  assert(root_vec_out->is_root());
  auto nodes_vec_in = vec_in_->Bfs();
  auto nodes_vec_out = vec_out_->Bfs();
  assert(nodes_vec_in.size());
  assert(nodes_vec_out.size());

  // Determine whether the input and output vector coincide.
  symmetric_ = (nodes_vec_in.size() == nodes_vec_out.size());
  if (symmetric_)
    for (size_t i = 0; i < nodes_vec_in.size(); ++i)
      if (nodes_vec_in[i]->node() != nodes_vec_out[i]->node()) {
        symmetric_ = false;
        break;
      }

  // If this applicator is symmetric, there is not a lot to do.
  if (symmetric_) {
    triang_ = std::make_shared<TriangulationView>(nodes_vec_in);
    nodes_vec_in_ =
        std::make_shared<std::vector<I_in*>>(std::move(nodes_vec_in));
    nodes_vec_out_ =
        std::make_shared<std::vector<I_out*>>(std::move(nodes_vec_out));
  } else {
    // This operator is not symmetric, calculate a union.
    vec_union_ =
        std::make_unique<datastructures::TreeVector<HierarchicalBasisFn>>(
            vec_in_->node());
    vec_union_->root()->Union(vec_in_);
    vec_union_->root()->Union(vec_out_);
    nodes_vec_union_ = std::make_shared<
        std::vector<datastructures::NodeVector<HierarchicalBasisFn>*>>(
        vec_union_->Bfs());
    triang_ = std::make_shared<TriangulationView>(*nodes_vec_union_);
  }
  operator_ = std::make_shared<Operator>(*triang_, opts);
}

template <typename Operator, typename I_in, typename I_out>
void BilinearForm<Operator, I_in, I_out>::Apply() {
  if (symmetric_) {
    // Apply the operator in SS.
    auto v = ToVector(*nodes_vec_in_);
    operator_->Apply(v);
    FromVector(*nodes_vec_out_, v);
    return;
  }

  // Not symmetric, we must do some magic tricks.
  auto lambda_copy = [](const auto& new_node, const auto& old_node) {
    new_node->set_value(old_node->value());
  };
  vec_union_->Reset();
  vec_union_->root()->Union(vec_in_, datastructures::func_false, lambda_copy);

  // Apply the operator in SS.
  auto v = ToVector(*nodes_vec_union_);
  operator_->Apply(v);
  FromVector(*nodes_vec_union_, v);
  // Copy the results from the union vector back to the output vector.
  vec_out_->Union(vec_union_->root(), datastructures::func_false, lambda_copy);
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
    if (!operator_->DirichletBoundary() ||
        !nodes[i]->node()->on_domain_boundary())
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
