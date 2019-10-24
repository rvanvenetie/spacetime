#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "tree.hpp"

struct Vertex;
using VertexPtr = std::shared_ptr<Vertex>;
using VectorVertexPtr = std::vector<VertexPtr>;
template <size_t N>
using ArrayVertexPtr = std::array<VertexPtr, N>;

class Element2D;
using Element2DPtr = std::shared_ptr<Element2D>;
using VectorElement2DPtr = std::vector<Element2DPtr>;
template <size_t N>
using ArrayElement2DPtr = std::array<Element2DPtr, N>;
std::ostream &operator<<(std::ostream &os, const Element2D &elem);

struct Vertex {
  double x, y;
  bool on_domain_boundary;
  VectorVertexPtr parents;
  VectorVertexPtr children;
  Vertex(double x, double y, bool on_domain_boundary, VectorVertexPtr parents,
         VectorVertexPtr children)
      : x(x),
        y(y),
        on_domain_boundary(on_domain_boundary),
        parents(parents),
        children(children) {}
};

std::ostream &operator<<(std::ostream &os, const Vertex &vertex) {
  os << "V(" << vertex.x << ", " << vertex.y << ")";
  return os;
}

class InitialTriangulation;

class Element2D : public Node<Element2D> {
 protected:
  double area_;
  VectorVertexPtr vertices_;

  VertexPtr create_new_vertex(Element2DPtr nbr = nullptr) {
    assert(is_leaf());
    VectorVertexPtr vertex_parents{newest_vertex()};
    if (nbr) {
      assert(nbr->edge(0) == reversed_edge(0));
      vertex_parents.emplace_back(nbr->newest_vertex());
    }

    ArrayVertexPtr<2> godparents{{vertices_[1], vertices_[2]}};
    auto new_vertex = std::make_shared<Vertex>(
        (godparents[0]->x + godparents[1]->x) / 2,
        (godparents[0]->y + godparents[1]->y) / 2, nbr != nullptr,
        vertex_parents, VectorVertexPtr{});
    for (auto vertex_parent : vertex_parents)
      vertex_parent->children.push_back(vertex_parent);
    return new_vertex;
  }

  ArrayElement2DPtr<2> bisect(VertexPtr new_vertex = nullptr) {
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
    children = {{child1, child2}};
    child1->neighbours = {{neighbours[2], nullptr, child2}};
    child2->neighbours = {{neighbours[1], child1, nullptr}};

    assert(child1->edge(2) == child2->reversed_edge(1));
    // new_vertex->patch.append(child1, child2);

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

  void bisect_with_nbr() {
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

 public:
  ArrayElement2DPtr<3> neighbours;
  bool is_full() const { return children.size() == 2; }
  bool is_leaf() const { return children.size() == 0; }
  Element2DPtr parent() const { return parents_[0]; }

  double area() const { return area_; }
  const VectorVertexPtr &vertices() const { return vertices_; }
  VertexPtr newest_vertex() const { return vertices_[0]; }
  ArrayVertexPtr<2> edge(int i) const {
    assert(0 <= i && i <= 2);
    return {{vertices_[(i + 1) % 3], vertices_[(i + 2) % 3]}};
  }
  ArrayVertexPtr<2> reversed_edge(int i) const {
    assert(0 <= i && i <= 2);
    return {{vertices_[(i + 2) % 3], vertices_[(i + 1) % 3]}};
  }
  Element2D() : Node(), area_(-1) {}

  explicit Element2D(Element2DPtr parent, const VectorVertexPtr &vertices,
                     double area)
      : Node({parent}), area_(area), vertices_(vertices) {}

  explicit Element2D(Element2DPtr parent, const VectorVertexPtr &vertices)
      : Element2D(parent, vertices, parent->area() / 2.0) {}

  const VectorElement2DPtr &refine() {
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
    return children;
  }
};
std::ostream &operator<<(std::ostream &os, const Element2D &elem) {
  os << "Element2D(" << elem.level() << ", (";
  for (auto vertex : elem.vertices()) {
    if (vertex != *elem.vertices().begin()) os << ", ";
    os << *vertex;
  }
  os << "))";
  return os;
}

class InitialTriangulation {
 public:
  VertexPtr vertex_meta_root;
  Element2DPtr elem_meta_root;
  InitialTriangulation(const std::vector<std::array<double, 2>> &vertices,
                       const std::vector<std::array<int, 3>> &elements)
      : vertex_meta_root(std::make_shared<Vertex>(
            -1, -1, false, VectorVertexPtr{}, VectorVertexPtr{})),
        elem_meta_root(std::make_shared<Element2D>()) {
    // Convenient aliases
    auto &vertex_roots = vertex_meta_root->children;
    auto &element_roots = elem_meta_root->children;

    for (auto vertex : vertices) {
      vertex_roots.push_back(std::make_shared<Vertex>(
          vertex[0], vertex[1], false, VectorVertexPtr{vertex_meta_root},
          VectorVertexPtr{}));
    }

    for (auto element : elements) {
      double elem_area =
          0.5 *
          std::abs((vertices[element[0]][0] - vertices[element[2]][0]) *
                       (vertices[element[1]][1] - vertices[element[0]][1]) -
                   (vertices[element[0]][0] - vertices[element[1]][0]) *
                       (vertices[element[2]][1] - vertices[element[0]][1]));
      element_roots.push_back(std::make_shared<Element2D>(
          elem_meta_root,
          VectorVertexPtr{vertex_roots[element[0]], vertex_roots[element[1]],
                          vertex_roots[element[2]]},
          elem_area));
    }

    for (int i = 0; i < elements.size(); ++i) {
      for (int j = i + 1; j < elements.size(); ++j) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            if (element_roots[i]->edge(k) ==
                element_roots[j]->reversed_edge(l)) {
              element_roots[i]->neighbours[k] = element_roots[j];
              element_roots[j]->neighbours[l] = element_roots[i];
            }
          }
        }
      }
    }
    for (auto element : element_roots) {
      for (int i = 0; i < 3; ++i)
        if (!element->neighbours[i]) {
          for (auto v : element->edge(i)) {
            v->on_domain_boundary = true;
          }
        }
    }
  }

  static InitialTriangulation unit_square() {
    std::vector<std::array<double, 2>> vertices = {
        {0, 0}, {1, 1}, {1, 0}, {0, 1}};
    std::vector<std::array<int, 3>> elements = {{0, 2, 3}, {1, 3, 2}};
    return InitialTriangulation(vertices, elements);
  }
};
