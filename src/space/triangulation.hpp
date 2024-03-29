#pragma once
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "datastructures/boost.hpp"
#include "datastructures/tree.hpp"

namespace space {
// Forward some class names.
class Vertex;
class Element2D;
class InitialTriangulation;
class HierarchicalBasisFn;
}  // namespace space

namespace datastructures {
template <>
struct NodeTrait<space::Vertex> {
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 4;
};
template <>
struct NodeTrait<space::Element2D> {
  static constexpr size_t N_parents = 1;
  static constexpr size_t N_children = 2;
};
}  // namespace datastructures

namespace space {
class Vertex : public datastructures::Node<Vertex> {
 public:
  const double x, y;
  bool on_domain_boundary;
  SmallVector<Element2D *, 4> patch;

  // This are the vertices that are bisected to create the current vertex.
  StaticVector<Vertex *, 2> godparents;
  StaticVector<Element2D *, 2> parents_element;

  // Constructor given parents.
  Vertex(const std::vector<Vertex *> &parents, double x, double y,
         bool on_domain_boundary)
      : Node(parents), x(x), y(y), on_domain_boundary(on_domain_boundary) {}

  friend std::ostream &operator<<(std::ostream &os, const Vertex &vertex) {
    os << "V(" << vertex.x << ", " << vertex.y << ")";
    return os;
  }

  bool Refine();
  bool is_full() const;

  // This returns the associated HierarchicalBasisFn.
  HierarchicalBasisFn *RefineHierarchicalBasisFn();

 protected:
  // Protected constructor for creating a metaroot.
  Vertex(Deque<Vertex> *container)
      : Node(container), x(NAN), y(NAN), on_domain_boundary(false) {}

  // There is a mapping between a vertex and a basis function.
  HierarchicalBasisFn *phi_ = nullptr;

  friend datastructures::Tree<Vertex>;
  friend Element2D;
  friend InitialTriangulation;
};

class Element2D : public datastructures::BinaryNode<Element2D> {
 public:
  std::array<Element2D *, 3> neighbours{nullptr};

  // Constructors given the parent.
  explicit Element2D(Element2D *parent, const std::array<Vertex *, 3> &vertices,
                     double area);
  explicit Element2D(Element2D *parent, const std::array<Vertex *, 3> &vertices)
      : Element2D(parent, vertices, parent->area() / 2.0) {}

  double area() const { return area_; }
  const std::array<Vertex *, 3> &vertices() const { return vertices_; }
  Vertex *newest_vertex() const { return vertices_[0]; }
  std::array<Vertex *, 2> edge(int i) const;
  std::array<Vertex *, 2> reversed_edge(int i) const;

  inline bool TouchesDomainBoundary() const {
    for (auto nbr : neighbours)
      if (!nbr) return true;
    return false;
  }

  Eigen::Vector3d BarycentricCoordinates(double x, double y) const;
  std::pair<double, double> GlobalCoordinates(double bary2, double bary3) const;
  const Eigen::Matrix3d &StiffnessMatrix() const { return stiff_mat_; }

  bool Refine();

  friend std::ostream &operator<<(std::ostream &os, const Element2D &elem) {
    if (elem.is_metaroot()) {
      os << "Element2D(MR)";
    } else {
      os << "Element2D(" << elem.level() << ", (";
      for (const auto &vertex : elem.vertices()) {
        if (vertex != *elem.vertices().begin()) os << ", ";
        os << *vertex;
      }
      os << "))";
    }
    return os;
  }

 protected:
  double area_;
  std::array<Vertex *, 3> vertices_;
  Eigen::Matrix3d stiff_mat_;

  // Protected constructor for creating a metaroot.
  Element2D(Deque<Element2D> *container) : BinaryNode(container), area_(-1) {}

  // Refinement methods.
  Vertex *CreateNewVertex(Element2D *nbr = nullptr);
  std::array<Element2D *, 2> Bisect(Vertex *new_vertex = nullptr);
  void BisectWithNbr();

  friend datastructures::Tree<Element2D>;
  friend InitialTriangulation;
};

};  // namespace space
