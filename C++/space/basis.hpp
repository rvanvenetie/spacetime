#pragma once
#include <Eigen/Dense>

#include "../datastructures/tree.hpp"
#include "triangulation.hpp"

namespace space {

class HierarchicalBasisFn : public datastructures::Node<HierarchicalBasisFn> {
 public:
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 4;

  HierarchicalBasisFn(const std::vector<HierarchicalBasisFn *> &parents,
                      Vertex *vertex)
      : Node(parents), vertex_(vertex) {}

  bool Refine();
  bool is_full() const;
  Vertex *vertex() const { return vertex_; }
  const std::vector<Element2D *> &support() const { return vertex_->patch; }

  double Eval(double x, double y) const;
  Eigen::Vector2d EvalGrad(double x, double y) const;

  friend std::ostream &operator<<(std::ostream &os,
                                  const HierarchicalBasisFn &fn) {
    os << "HBF(" << fn.vertex()->x << ", " << fn.vertex()->y << ")";
    return os;
  }

 private:
  Vertex *vertex_;

  // Protected constructor for creating a metaroot.
  HierarchicalBasisFn(Vertex *vertex) : Node(), vertex_(vertex) {
    assert(vertex->is_metaroot());
  }

  friend datastructures::Tree<HierarchicalBasisFn>;
  friend InitialTriangulation;
  friend Vertex;
};

}  // namespace space
