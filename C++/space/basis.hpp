#pragma once
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

  bool Refine() {
    if (is_full()) return false;
    for (auto &elem : vertex_->patch) {
      elem->Refine();
      assert(elem->children().size() == 2);
      elem->children()[0]->newest_vertex()->RefineHierarchicalBasisFn();
    }
    assert(is_full());
    return true;
  }

  bool is_full() const {
    if (is_metaroot()) return vertex_->children().size() == children().size();
    for (auto &elem : vertex_->patch)
      if (!elem->is_full()) return false;
    return true;
  }

  Vertex *vertex() const { return vertex_; }

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
