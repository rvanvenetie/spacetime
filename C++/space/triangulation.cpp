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

const VectorElement2DPtr &Element2D::refine() {
  if (!is_full()) {
    auto nbr = neighbours[0];
    if (!nbr) {  // Refinement edge of `elem` is on domain boundary
      bisect();
    } else if (nbr->edge(0) != reversed_edge(0)) {
      nbr->refine();
      return refine();
    } else {
      bisect_with_nbr();
    }
  }
  return children_;
}

VertexPtr Element2D::create_new_vertex(Element2DPtr nbr) {
  assert(is_leaf());
  VectorVertexPtr vertex_parents{newest_vertex()};
  if (nbr) {
    assert(nbr->edge(0) == reversed_edge(0));
    vertex_parents.emplace_back(nbr->newest_vertex());
  }

  ArrayVertexPtr<2> godparents{{vertices_[1], vertices_[2]}};
  auto new_vertex =
      std::make_shared<Vertex>((godparents[0]->x + godparents[1]->x) / 2,
                               (godparents[0]->y + godparents[1]->y) / 2,
                               nbr == nullptr, vertex_parents);
  for (auto vertex_parent : vertex_parents)
    vertex_parent->children().push_back(new_vertex);
  return new_vertex;
}

ArrayElement2DPtr<2> Element2D::bisect(VertexPtr new_vertex) {
  assert(is_leaf());
  if (!new_vertex) {
    new_vertex = create_new_vertex();
  }
  auto child1 = std::make_shared<Element2D>(
      shared_from_this(),
      VectorVertexPtr{{new_vertex, vertices_[0], vertices_[1]}});
  auto child2 = std::make_shared<Element2D>(
      shared_from_this(),
      VectorVertexPtr{{new_vertex, vertices_[2], vertices_[0]}});
  children_ = {{child1, child2}};
  child1->neighbours = {{neighbours[2], nullptr, child2}};
  child2->neighbours = {{neighbours[1], child1, nullptr}};

  assert(child1->edge(2) == child2->reversed_edge(1));
  new_vertex->patch.push_back(child1);
  new_vertex->patch.push_back(child2);

  if (neighbours[2]) {
    for (int i = 0; i < 3; ++i) {
      if (neighbours[2]->neighbours[i] == shared_from_this()) {
        neighbours[2]->neighbours[i] = child1;
      }
    }
  }
  if (neighbours[1]) {
    for (int i = 0; i < 3; ++i) {
      if (neighbours[1]->neighbours[i] == shared_from_this()) {
        neighbours[1]->neighbours[i] = child2;
      }
    }
  }
  return {{child1, child2}};
}

void Element2D::bisect_with_nbr() {
  auto nbr = neighbours[0];
  assert(edge(0) == nbr->reversed_edge(0));

  auto new_vertex = create_new_vertex(nbr);
  auto children0 = bisect(new_vertex);
  auto children1 = nbr->bisect(new_vertex);

  children0[0]->neighbours[1] = children1[1];
  children0[1]->neighbours[2] = children1[0];
  children1[0]->neighbours[1] = children0[1];
  children1[1]->neighbours[2] = children0[0];
}

InitialTriangulation::InitialTriangulation(
    const std::vector<std::array<double, 2>> &vertices,
    const std::vector<std::array<int, 3>> &elements)
    : vertex_meta_root(Vertex::CreateMetaroot()),
      elem_meta_root(Element2D::CreateMetaroot()) {
  // Convenient aliases
  auto &vertex_roots = vertex_meta_root->children();
  auto &element_roots = elem_meta_root->children();

  for (auto vertex : vertices) {
    vertex_roots.push_back(std::make_shared<Vertex>(
        vertex[0], vertex[1], false, VectorVertexPtr{vertex_meta_root}));
  }

  for (auto element : elements) {
    double elem_area =
        0.5 * std::abs((vertices[element[0]][0] - vertices[element[2]][0]) *
                           (vertices[element[1]][1] - vertices[element[0]][1]) -
                       (vertices[element[0]][0] - vertices[element[1]][0]) *
                           (vertices[element[2]][1] - vertices[element[0]][1]));
    element_roots.push_back(std::make_shared<Element2D>(
        elem_meta_root,
        VectorVertexPtr{vertex_roots[element[0]], vertex_roots[element[1]],
                        vertex_roots[element[2]]},
        elem_area));
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
  for (auto element : element_roots) {
    for (auto vertex : element->vertices()) {
      vertex->patch.push_back(element);
    }
  }

  // Determine vertices that are on the boundary.
  for (auto element : element_roots) {
    for (int i = 0; i < 3; ++i)
      if (!element->neighbours[i]) {
        for (auto v : element->edge(i)) {
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
