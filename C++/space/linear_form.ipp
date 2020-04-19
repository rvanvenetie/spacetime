#pragma once

#include "../tools/integration.hpp"

namespace space {
namespace {
double EvalHatFn(double x, double y, Element2D *elem, size_t i) {
  auto bary = elem->BarycentricCoordinates(x, y);
  assert((bary.array() >= 0).all());
  return bary[i];
}
}  // namespace

std::array<double, 3> QuadratureFunctional::Eval(Element2D *elem) const {
  std::array<double, 3> result;
  for (size_t i = 0; i < 3; i++)
    result[i] = tools::Integrate2D(
        [&](double x, double y) { return f_(x, y) * EvalHatFn(x, y, elem, i); },
        *elem, order_ + 1);
  return result;
}

template <typename I>
void LinearForm::Apply(I *root) {
  assert(root->is_root());
  auto triang = TriangulationView(root->Bfs());
  const auto &vertices = triang.vertices();
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(vertices.size());
  for (const auto &elem : triang.elements()) {
    if (!elem->is_leaf()) continue;
    auto &Vids = elem->vertices_view_idx_;
    auto eval = functional_->Eval(elem->node());
    for (size_t i = 0; i < 3; ++i) vec[Vids[i]] += eval[i];
  }

  int vi = triang.vertices().size() - 1;
  for (; vi >= triang.InitialVertices(); --vi)
    for (auto gp : triang.history(vi)[0]->RefinementEdge())
      vec[gp] = vec[gp] + 0.5 * vec[vi];

  if (dirichlet_boundary_)
    for (int i = 0; i < vertices.size(); ++i)
      if (vertices[i]->on_domain_boundary) vec[i] = 0;

  assert(root->Bfs().size() == vec.size());
  root->FromVector(vec);
}

}  // namespace space
