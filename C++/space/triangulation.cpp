#include "triangulation.hpp"

namespace space {

ArrayVertexPtr<2> Element2D::edge(int i) const {
  assert(0 <= i && i <= 2);
  return {{vertices_[(i + 1) % 3], vertices_[(i + 2) % 3]}};
}

ArrayVertexPtr<2> Element2D::reversed_edge(int i) const {
  assert(0 <= i && i <= 2);
  return {{vertices_[(i + 2) % 3], vertices_[(i + 1) % 3]}};
}

const VectorElement2DPtr &Element2DTree::Refine(Element2DPtr elem) {
  if (!elem->is_full()) {
    auto nbr = elem->neighbours[0];
    if (!nbr) {
      // Refinement edge of `elem` is on domain boundary
      auto new_vertex = CreateNewVertex(elem, nbr);
      Bisect(elem, new_vertex);
    } else if (nbr->edge(0) != elem->reversed_edge(0)) {
      Refine(nbr);
      return Refine(elem);
    } else {
      BisectWithNbr(elem, nbr);
    }
  }
  return elem->children_;
}

VertexPtr Element2DTree::CreateNewVertex(Element2DPtr elem, Element2DPtr nbr) {
  assert(elem->is_leaf());
  VectorVertexPtr vertex_parents{elem->newest_vertex()};
  if (nbr) {
    assert(nbr->edge(0) == elem->reversed_edge(0));
    vertex_parents.emplace_back(nbr->newest_vertex());
  }

  ArrayVertexPtr<2> godparents{{elem->vertices_[1], elem->vertices_[2]}};
  auto new_vertex =
      vertex_tree.emplace_back((godparents[0]->x + godparents[1]->x) / 2,
                               (godparents[0]->y + godparents[1]->y) / 2,
                               nbr == nullptr, vertex_parents);
  for (const auto &vertex_parent : vertex_parents)
    vertex_parent->children().push_back(new_vertex);
  return new_vertex;
}

ArrayElement2DPtr<2> Element2DTree::Bisect(Element2DPtr elem,
                                           VertexPtr new_vertex) {
  assert(elem->is_leaf());
  auto child1 = emplace_back(
      elem,
      ArrayVertexPtr<3>{{new_vertex, elem->vertices_[0], elem->vertices_[1]}});
  auto child2 = emplace_back(
      elem,
      ArrayVertexPtr<3>{{new_vertex, elem->vertices_[2], elem->vertices_[0]}});
  elem->children_ = {{child1, child2}};
  child1->neighbours = {{elem->neighbours[2], nullptr, child2}};
  child2->neighbours = {{elem->neighbours[1], child1, nullptr}};

  assert(child1->edge(2) == child2->reversed_edge(1));
  new_vertex->patch.push_back(child1);
  new_vertex->patch.push_back(child2);

  if (elem->neighbours[2]) {
    for (int i = 0; i < 3; ++i) {
      if (elem->neighbours[2]->neighbours[i] == elem) {
        elem->neighbours[2]->neighbours[i] = child1;
      }
    }
  }
  if (elem->neighbours[1]) {
    for (int i = 0; i < 3; ++i) {
      if (elem->neighbours[1]->neighbours[i] == elem) {
        elem->neighbours[1]->neighbours[i] = child2;
      }
    }
  }
  return {{child1, child2}};
}

void Element2DTree::BisectWithNbr(Element2DPtr elem, Element2DPtr nbr) {
  assert(elem->edge(0) == nbr->reversed_edge(0));

  auto new_vertex = CreateNewVertex(elem, nbr);
  auto children0 = Bisect(elem, new_vertex);
  auto children1 = Bisect(nbr, new_vertex);

  children0[0]->neighbours[1] = children1[1];
  children0[1]->neighbours[2] = children1[0];
  children1[0]->neighbours[1] = children0[1];
  children1[1]->neighbours[2] = children0[0];
}

InitialTriangulation::InitialTriangulation(
    const std::vector<std::array<double, 2>> &vertices,
    const std::vector<std::array<int, 3>> &elements)
    : vertex_tree(),
      elem_tree(vertex_tree),
      vertex_meta_root(vertex_tree.meta_root),
      elem_meta_root(elem_tree.meta_root) {
  // Convenient aliases
  auto &vertex_roots = vertex_tree.meta_root->children();
  auto &element_roots = elem_tree.meta_root->children();

  for (const auto &vertex : vertices) {
    auto vertex_ptr = vertex_tree.emplace_back(
        vertex[0], vertex[1], false, VectorVertexPtr{vertex_tree.meta_root});
    vertex_roots.push_back(vertex_ptr);
  }

  for (const auto &element : elements) {
    double elem_area =
        0.5 * std::abs((vertices[element[0]][0] - vertices[element[2]][0]) *
                           (vertices[element[1]][1] - vertices[element[0]][1]) -
                       (vertices[element[0]][0] - vertices[element[1]][0]) *
                           (vertices[element[2]][1] - vertices[element[0]][1]));
    auto elem_ptr = elem_tree.emplace_back(
        elem_tree.meta_root,
        ArrayVertexPtr<3>{vertex_roots[element[0]], vertex_roots[element[1]],
                          vertex_roots[element[2]]},
        elem_area);
    element_roots.push_back(elem_ptr);
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
};  // namespace space
