#pragma once

#include "triangulation_view.hpp"

namespace space {
template <typename I>
void LinearForm::Apply(I *root) {
  assert(root->is_root());
  auto triang = TriangulationView(root->Bfs());
  const auto &vertices = triang.vertices();
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(vertices.size());
  for (const auto &[elem, Vids] : triang.element_leaves()) {
    auto eval = functional_->Eval(elem);
    for (size_t i = 0; i < 3; ++i) vec[Vids[i]] += eval[i];
  }

  int vi = triang.vertices().size() - 1;
  for (; vi >= triang.InitialVertices(); --vi)
    for (auto gp : triang.Godparents(vi)) vec[gp] = vec[gp] + 0.5 * vec[vi];

  if (dirichlet_boundary_)
    for (int i = 0; i < vertices.size(); ++i)
      if (vertices[i]->on_domain_boundary) vec[i] = 0;

  assert(root->Bfs().size() == vec.size());
  root->FromVector(vec);
}
}  // namespace space
