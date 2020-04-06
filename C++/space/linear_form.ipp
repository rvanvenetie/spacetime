#pragma once

#include <boost/range/adaptor/reversed.hpp>

#include "../tools/integration.hpp"
#include "linear_form.hpp"
#include "triangulation_view.hpp"

namespace space {
namespace {
double EvalHatFn(double x, double y, Element2D *elem, size_t i) {
  auto bary = elem->BarycentricCoordinates(x, y);
  assert((bary.array() >= 0).all());
  return bary[i];
}
}  // namespace

template <size_t order, class F, typename I>
void ApplyQuadrature(const F &f, I *root, bool dirichlet_boundary) {
  assert(root->is_root());
  auto triang = TriangulationView(root->Bfs());
  const auto &vertices = triang.vertices();
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(vertices.size());
  for (const auto &elem : triang.elements()) {
    if (!elem->is_leaf()) continue;
    auto &Vids = elem->vertices_view_idx_;
    for (size_t i = 0; i < 3; ++i)
      vec[Vids[i]] +=
          tools::IntegrationRule</*dim*/ 2, /*order*/ order + 1>::Integrate(
              [&](double x, double y) {
                return f(x, y) * EvalHatFn(x, y, elem->node(), i);
              },
              *elem->node());
  }

  for (int vi = triang.vertices().size(); vi >= 0; --vi) {
    const auto &hist = triang.history(vi);
    if (hist.empty()) continue;  // Vertex on initial mesh.
    for (auto gp : hist.at(0)->RefinementEdge())
      vec[gp] = vec[gp] + 0.5 * vec[vi];
  }

  if (dirichlet_boundary)
    for (int i = 0; i < vertices.size(); ++i)
      if (vertices[i]->on_domain_boundary) vec[i] = 0;

  assert(root->Bfs().size() == vec.size());
  root->FromVector(vec);
}
}  // namespace space
