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

bool Element2D::Refine() {
  if (!is_full()) {
    auto nbr = neighbours[0];
    if (!nbr) {  // Refinement edge of `elem` is on domain boundary
      Bisect();
    } else if (nbr->edge(0) != reversed_edge(0)) {
      nbr->Refine();
      return Refine();
    } else {
      BisectWithNbr();
    }
    return true;
  }
  return false;
}

VertexPtr Element2D::CreateNewVertex(Element2DPtr nbr) {
  assert(is_leaf());
  std::vector<Vertex *> vertex_parents{newest_vertex()};
  if (nbr) {
    assert(nbr->edge(0) == reversed_edge(0));
    vertex_parents.emplace_back(nbr->newest_vertex());
  }

  ArrayVertexPtr<2> godparents{{vertices_[1], vertices_[2]}};
  auto new_vertex = vertex_parents[0]->make_child(
      /* parents */ vertex_parents,
      /* x */ (godparents[0]->x + godparents[1]->x) / 2,
      /* y */ (godparents[0]->y + godparents[1]->y) / 2,
      /* on_domain_boundary */ nbr == nullptr);
  vertex_parents[0]->children_own_.emplace_back(new_vertex);
  if (vertex_parents.size() == 2)
    vertex_parents[1]->children_.emplace_back(new_vertex);
  return new_vertex;
}

ArrayElement2DPtr<2> Element2D::Bisect(VertexPtr new_vertex) {
  assert(is_leaf());
  if (!new_vertex) {
    new_vertex = CreateNewVertex();
  }
  auto child1 = make_child(/* parent */ this, /* vertices */ ArrayVertexPtr<3>{
                               {new_vertex, vertices_[0], vertices_[1]}});
  auto child2 = make_child(/* parent */ this, /* vertices */ ArrayVertexPtr<3>{
                               {new_vertex, vertices_[2], vertices_[0]}});
  child1->neighbours = {{neighbours[2], nullptr, child2}};
  child2->neighbours = {{neighbours[1], child1, nullptr}};

  assert(child1->edge(2) == child2->reversed_edge(1));
  new_vertex->patch.push_back(child1);
  new_vertex->patch.push_back(child2);

  if (neighbours[2]) {
    for (int i = 0; i < 3; ++i) {
      if (neighbours[2]->neighbours[i] == this) {
        neighbours[2]->neighbours[i] = child1;
      }
    }
  }
  if (neighbours[1]) {
    for (int i = 0; i < 3; ++i) {
      if (neighbours[1]->neighbours[i] == this) {
        neighbours[1]->neighbours[i] = child2;
      }
    }
  }
  return {{child1, child2}};
}

void Element2D::BisectWithNbr() {
  auto nbr = neighbours[0];
  assert(edge(0) == nbr->reversed_edge(0));

  auto new_vertex = CreateNewVertex(nbr);
  auto children0 = Bisect(new_vertex);
  auto children1 = nbr->Bisect(new_vertex);

  children0[0]->neighbours[1] = children1[1];
  children0[1]->neighbours[2] = children1[0];
  children1[0]->neighbours[1] = children0[1];
  children1[1]->neighbours[2] = children0[0];
}

InitialTriangulation::InitialTriangulation(
    const std::vector<std::array<double, 2>> &vertices,
    const std::vector<std::array<int, 3>> &elements)
    : vertex_tree(),
      elem_tree(),
      vertex_meta_root(vertex_tree.meta_root.get()),
      elem_meta_root(elem_tree.meta_root.get()) {
  // Convenient aliases.
  auto &vertex_roots = vertex_meta_root->children();
  auto &element_roots = elem_meta_root->children();
  for (const auto &vertex : vertices) {
    vertex_meta_root->make_child(
        /* parents */ std::vector<Vertex *>{vertex_meta_root},
        /* x */ vertex[0], /* y */ vertex[1], /* on_domain_boundary */ false);
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
        ArrayVertexPtr<3>{vertex_roots[element[0]], vertex_roots[element[1]],
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
};  // namespace space
