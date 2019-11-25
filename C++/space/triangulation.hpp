#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "datastructures/tree.hpp"

namespace space {

// Forward some class names.
class Vertex;
class VertexTree;
class Element2D;
class Element2DTree;

// Define some aliases
using VertexPtr = Vertex *;
using VectorVertexPtr = std::vector<VertexPtr>;
template <size_t N>
using ArrayVertexPtr = std::array<VertexPtr, N>;

using Element2DPtr = Element2D *;
using VectorElement2DPtr = std::vector<Element2DPtr>;
template <size_t N>
using ArrayElement2DPtr = std::array<Element2DPtr, N>;

class Vertex : public datastructures::Node<Vertex> {
 public:
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 4;

  const double x, y;
  bool on_domain_boundary;
  VectorElement2DPtr patch;

  // Constructor for metaroot
  Vertex() : Node(), x(NAN), y(NAN), on_domain_boundary(false) {}

  // Constructor given parents.
  Vertex(double x, double y, bool on_domain_boundary,
         const VectorVertexPtr &parents)
      : Node(parents), x(x), y(y), on_domain_boundary(on_domain_boundary) {}

  Vertex(const Vertex &) = delete;

  friend std::ostream &operator<<(std::ostream &os, const Vertex &vertex) {
    os << "V(" << vertex.x << ", " << vertex.y << ")";
    return os;
  }

 protected:
  friend datastructures::Tree<Vertex>;
};

class VertexTree : public datastructures::Tree<Vertex> {
 public:
  using datastructures::Tree<Vertex>::Tree;

  const VectorVertexPtr &Refine(VertexPtr vertex) final { assert(false); }
};

class Element2D : public datastructures::BinaryNode<Element2D> {
 public:
  ArrayElement2DPtr<3> neighbours = {};

  // Constructor for creating a metaroot.
  Element2D() : BinaryNode(), area_(-1) {}

  // Constructors given the parent.
  explicit Element2D(Element2DPtr parent, const ArrayVertexPtr<3> &vertices,
                     double area)
      : BinaryNode(parent), area_(area), vertices_(vertices) {}
  explicit Element2D(Element2DPtr parent, const ArrayVertexPtr<3> &vertices)
      : Element2D(parent, vertices, parent->area() / 2.0) {}

  Element2D(const Element2D &) = delete;

  double area() const { return area_; }
  const ArrayVertexPtr<3> &vertices() const { return vertices_; }

  VertexPtr newest_vertex() const { return vertices_[0]; }
  ArrayVertexPtr<2> edge(int i) const;
  ArrayVertexPtr<2> reversed_edge(int i) const;

  friend std::ostream &operator<<(std::ostream &os, const Element2D &elem) {
    os << "Element2D(" << elem.level() << ", (";
    for (const auto &vertex : elem.vertices()) {
      if (vertex != *elem.vertices().begin()) os << ", ";
      os << *vertex;
    }
    os << "))";
    return os;
  }

 protected:
  double area_;
  ArrayVertexPtr<3> vertices_;

  friend Element2DTree;
  friend datastructures::Tree<Element2D>;
};

class Element2DTree : public datastructures::Tree<Element2D> {
 public:
  const VectorElement2DPtr &Refine(Element2DPtr elem) final;
  explicit Element2DTree(VertexTree &vertex_tree)
      : datastructures::Tree<Element2D>(), vertex_tree(vertex_tree) {}

 protected:
  VertexTree &vertex_tree;

  VertexPtr CreateNewVertex(Element2DPtr elem, Element2DPtr nbr);
  ArrayElement2DPtr<2> Bisect(Element2DPtr elem, VertexPtr new_vertex);
  void BisectWithNbr(Element2DPtr elem, Element2DPtr nbr);
};

class InitialTriangulation {
 public:
  VertexTree vertex_tree;
  Element2DTree elem_tree;

  VertexPtr const vertex_meta_root;
  Element2DPtr const elem_meta_root;

  InitialTriangulation(const std::vector<std::array<double, 2>> &vertices,
                       const std::vector<std::array<int, 3>> &elements);
  static InitialTriangulation UnitSquare();
};
};  // namespace space
