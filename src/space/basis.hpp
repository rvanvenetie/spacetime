#pragma once
#include <Eigen/Dense>

#include "../datastructures/tree.hpp"
#include "triangulation.hpp"

namespace datastructures {
template <>
struct NodeTrait<space::HierarchicalBasisFn> {
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 4;
};
}  // namespace datastructures

namespace space {
class HierarchicalBasisFn : public datastructures::Node<HierarchicalBasisFn> {
 public:
  HierarchicalBasisFn(const std::vector<HierarchicalBasisFn *> &parents,
                      Vertex *vertex)
      : Node(parents), vertex_(vertex) {}

  bool Refine();
  bool is_full() const;
  Vertex *vertex() const { return vertex_; }
  const SmallVector<Element2D *, 4> &support() const { return vertex_->patch; }
  std::pair<double, double> center() const { return {vertex_->x, vertex_->y}; }
  inline bool on_domain_boundary() const { return vertex_->on_domain_boundary; }

  // Whether some element in the support touches the boundary.}
  inline bool TouchesDomainBoundary() const {
    for (auto elem : support())
      if (elem->TouchesDomainBoundary()) return true;
    return false;
  }

  double Volume() const;

  double Eval(double x, double y) const;
  bool Contains(double x, double y) const;
  Eigen::Vector2d EvalGrad(double x, double y) const;

  friend std::ostream &operator<<(std::ostream &os,
                                  const HierarchicalBasisFn &fn) {
    os << "HBF(" << fn.vertex()->x << ", " << fn.vertex()->y << ")";
    return os;
  }

 private:
  Vertex *vertex_;

  // Protected constructor for creating a metaroot.
  HierarchicalBasisFn(Deque<HierarchicalBasisFn> *container, Vertex *vertex)
      : Node(container), vertex_(vertex) {
    assert(vertex->is_metaroot());
  }

  friend datastructures::Tree<HierarchicalBasisFn>;
  friend InitialTriangulation;
  friend Vertex;
};

}  // namespace space
