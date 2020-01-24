#include "initial_triangulation.hpp"
namespace space {

InitialTriangulation::InitialTriangulation(
    const std::vector<std::array<double, 2>> &vertices,
    const std::vector<std::array<int, 3>> &elements)
    : vertex_tree(),
      elem_tree(),
      hierarch_basis_tree(new HierarchicalBasisFn(vertex_tree.meta_root.get())),
      vertex_meta_root(vertex_tree.meta_root.get()),
      elem_meta_root(elem_tree.meta_root.get()),
      hierarch_basis_meta_root(hierarch_basis_tree.meta_root.get()) {
  // Convenient aliases.
  auto &vertex_roots = vertex_meta_root->children();
  auto &element_roots = elem_meta_root->children();
  for (const auto &vertex : vertices) {
    auto child = vertex_meta_root->make_child(
        /* parents */ std::vector<Vertex *>{vertex_meta_root},
        /* x */ vertex[0], /* y */ vertex[1], /* on_domain_boundary */ false);
    child->phi_ = hierarch_basis_meta_root->make_child(
        /* parents */ std::vector{hierarch_basis_meta_root},
        /* vertex */ child);
  }

  for (const auto &element : elements) {
    double elem_area =
        0.5 * std::abs((vertices[element[0]][0] - vertices[element[2]][0]) *
                           (vertices[element[1]][1] - vertices[element[0]][1]) -
                       (vertices[element[0]][0] - vertices[element[1]][0]) *
                           (vertices[element[2]][1] - vertices[element[0]][1]));
    elem_meta_root->make_child(
        /* parent */ elem_meta_root,
        /* vertices */
        std::array<Vertex *, 3>{vertex_roots[element[0]],
                                vertex_roots[element[1]],
                                vertex_roots[element[2]]},
        /* area */ elem_area);
  }

  // Set neighbour information.
  for (int i = 0; i < elements.size(); ++i) {
    for (int j = i + 1; j < elements.size(); ++j) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
          if (element_roots[i]->edge(k) == element_roots[j]->reversed_edge(l)) {
            element_roots[i]->neighbours[k] = element_roots[j];
            element_roots[j]->neighbours[l] = element_roots[i];
          }
        }
      }
    }
  }

  // Determine patches for the vertices.
  for (const auto &element : element_roots) {
    for (const auto &vertex : element->vertices()) {
      vertex->patch.push_back(element);
    }
  }

  // Determine vertices that are on the boundary.
  for (const auto &element : element_roots) {
    for (int i = 0; i < 3; ++i)
      if (!element->neighbours[i]) {
        for (const auto &v : element->edge(i)) {
          v->on_domain_boundary = true;
        }
      }
  }
}

InitialTriangulation InitialTriangulation::UnitSquare() {
  std::vector<std::array<double, 2>> vertices = {
      {{0, 0}}, {{1, 1}}, {{1, 0}}, {{0, 1}}};
  std::vector<std::array<int, 3>> elements = {{{0, 2, 3}}, {{1, 3, 2}}};
  return InitialTriangulation(vertices, elements);
}
}  // namespace space
