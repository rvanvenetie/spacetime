#pragma once
#include "bilinear_form.hpp"

namespace space {
template <typename Operator, typename I_in, typename I_out>
BilinearForm<Operator, I_in, I_out>::BilinearForm(I_in* root_vec_in,
                                                  I_out* root_vec_out,
                                                  const OperatorOptions& opts)
    : vec_in_(root_vec_in), vec_out_(root_vec_out) {
  auto time_start = std::chrono::high_resolution_clock::now();
  assert(vec_in_->is_root());
  assert(vec_out_->is_root());
  nodes_vec_in_ = std::make_shared<std::vector<I_in*>>(vec_in_->Bfs());
  nodes_vec_out_ = std::make_shared<std::vector<I_out*>>(vec_out_->Bfs());
  const auto& nodes_vec_in = *nodes_vec_in_;
  const auto& nodes_vec_out = *nodes_vec_out_;
  assert(nodes_vec_in.size());
  assert(nodes_vec_out.size());

  // Determine the inclusion relation.
  if (nodes_vec_in.size() < nodes_vec_out.size())
    inclusion_type_ = Subset;  // vec_in < vec_out.
  else if (nodes_vec_in.size() > nodes_vec_out.size())
    inclusion_type_ = Superset;  // vec_in > vec_out.
  else {
    inclusion_type_ = Equal;  // vec_in == vec_out.

    // It might be the case that the ordering vectors differs, then
    // simply make this a Subset Type.
    for (size_t i = 0; i < nodes_vec_in.size(); ++i)
      if (nodes_vec_in[i]->node() != nodes_vec_out[i]->node()) {
        inclusion_type_ = InclusionType::Subset;
        break;
      }
  }

  switch (inclusion_type_) {
    case Subset:
      // Assert that vec_out_ is a refinement of vec_in_.
      // assert((vec_out_->Union(vec_in_), vec_out_->Bfs().size()) ==
      //       nodes_vec_out.size());
    case Equal:
      // For both equal and subset, make a triangulation based on output nodes.
      triang_ = std::make_shared<TriangulationView>(nodes_vec_out);
      break;
    case Superset:
      // Assert that vec_in is a refinement of vec_out.
      // assert((vec_out_->Union(vec_out_), vec_out_->Bfs().size()) ==
      //       nodes_vec_out.size());
      triang_ = std::make_shared<TriangulationView>(nodes_vec_in);
      break;
    default:
      assert(false);
  }

  operator_ = std::make_shared<Operator>(*triang_, opts);
  time_create_ = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - time_start);
}

template <typename Operator, typename I_in, typename I_out>
void BilinearForm<Operator, I_in, I_out>::Apply() {
  auto time_start = std::chrono::high_resolution_clock::now();
  num_apply_++;

  if (inclusion_type_ == InclusionType::Equal) {
    // vec_in == vec_out.
    auto v = ToVector(*nodes_vec_in_);
    operator_->Apply(v);
    FromVector(*nodes_vec_out_, v);
    time_apply_ += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - time_start);
    return;
  }

  // Not symmetric. But one is a refinement of the other, copy the
  // data correspondingly.
  auto lambda_copy = [](const auto& new_node, const auto& old_node) {
    new_node->set_value(old_node->value());
  };
  if (inclusion_type_ == InclusionType::Subset) {
    // vec_in < vec_out.

    // Abuse the output vector for storing the input temporarily.
    for (const auto& nv : *nodes_vec_out_) nv->set_value(0);
    size_t s = vec_out_->Union(vec_in_, datastructures::func_false, lambda_copy)
                   .size();
    assert(s == nodes_vec_in_->size() + 1);

    Eigen::VectorXd v_out = ToVector(*nodes_vec_out_);
    operator_->Apply(v_out);
    FromVector(*nodes_vec_out_, v_out);
  } else if (inclusion_type_ == InclusionType::Superset) {
    // vec_in > vec_out.
    Eigen::VectorXd v_in = ToVector(*nodes_vec_in_);
    Eigen::VectorXd v_out = v_in;
    operator_->Apply(v_out);

    // Abuse the input vector for storing the output temporarily.
    FromVector(*nodes_vec_in_, v_out);
    size_t s = vec_out_->Union(vec_in_, datastructures::func_false, lambda_copy)
                   .size();
    assert(s == nodes_vec_out_->size() + 1);
    FromVector(*nodes_vec_in_, v_in);
  }
  time_apply_ += std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - time_start);
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
