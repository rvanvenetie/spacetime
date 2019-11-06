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
class Element2D;
class InitialTriangulation;

// Define some aliases
using VertexPtr = std::shared_ptr<Vertex>;
using VectorVertexPtr = std::vector<VertexPtr>;
template <size_t N>
using ArrayVertexPtr = std::array<VertexPtr, N>;

using Element2DPtr = std::shared_ptr<Element2D>;
using VectorElement2DPtr = std::vector<Element2DPtr>;
template <size_t N>
using ArrayElement2DPtr = std::array<Element2DPtr, N>;

class Vertex : public datastructures::Node<Vertex> {
 public:
  const double x, y;
  bool on_domain_boundary;
  VectorElement2DPtr patch;
  Vertex(double x, double y, bool on_domain_boundary,
         const VectorVertexPtr &parents)
      : Node(parents), x(x), y(y), on_domain_boundary(on_domain_boundary) {}

  friend std::ostream &operator<<(std::ostream &os, const Vertex &vertex) {
    os << "V(" << vertex.x << ", " << vertex.y << ")";
    return os;
  }

 protected:
  // Protected constructor for creating a metaroot.
  Vertex() : Node(), x(NAN), y(NAN), on_domain_boundary(false) {}

  friend Element2D;
  friend InitialTriangulation;
};

class Element2D : public datastructures::BinaryNode<Element2D> {
 public:
  ArrayElement2DPtr<3> neighbours;

  // Constructors given the parent.
  explicit Element2D(Element2DPtr parent, const VectorVertexPtr &vertices)
      : Element2D(parent, vertices, parent->area() / 2.0) {}
  explicit Element2D(Element2DPtr parent, const VectorVertexPtr &vertices,
                     double area)
      : BinaryNode(parent), area_(area), vertices_(vertices) {}

  double area() const { return area_; }
  const VectorVertexPtr &vertices() const { return vertices_; }

  VertexPtr newest_vertex() const { return vertices_[0]; }
  ArrayVertexPtr<2> edge(int i) const;
  ArrayVertexPtr<2> reversed_edge(int i) const;

  const VectorElement2DPtr &refine();

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
  VectorVertexPtr vertices_;

  // Protected constructor for creating a metaroot.
  Element2D() : BinaryNode(), area_(-1) {}

  VertexPtr create_new_vertex(Element2DPtr nbr = nullptr);
  ArrayElement2DPtr<2> bisect(VertexPtr new_vertex = nullptr);
  void bisect_with_nbr();

  friend InitialTriangulation;
};

class InitialTriangulation {
 public:
  VertexPtr vertex_meta_root;
  Element2DPtr elem_meta_root;

  InitialTriangulation(const std::vector<std::array<double, 2>> &vertices,
                       const std::vector<std::array<int, 3>> &elements);
  static InitialTriangulation UnitSquare();
};
};  // namespace space
