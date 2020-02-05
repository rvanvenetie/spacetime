#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "../datastructures/multi_tree_vector.hpp"
#include "../datastructures/multi_tree_view.hpp"
#include "basis.hpp"
#include "triangulation.hpp"

namespace space {

class Element2DView
    : public datastructures::NodeViewBase<Element2DView, Element2D> {
 public:
  using datastructures::NodeViewBase<Element2DView, Element2D>::NodeViewBase;

  size_t NewestVertex() const { return vertices_view_idx_[0]; }
  std::array<size_t, 2> RefinementEdge() const {
    return {vertices_view_idx_[1], vertices_view_idx_[2]};
  }

  std::array<size_t, 3> vertices_view_idx_;
};

class TriangulationView {
 public:
  TriangulationView(std::vector<Vertex *> vertices);
  template <typename I>
  TriangulationView(const std::shared_ptr<I> &root)
      : TriangulationView(Transform(root)) {}
  TriangulationView(const datastructures::TreeView<Vertex> &vertex_view)
      : TriangulationView(Transform(vertex_view.root)) {}
  TriangulationView(
      const datastructures::TreeVector<HierarchicalBasisFn> &basis_vector)
      : TriangulationView(Transform(basis_vector.root)) {}

  const std::vector<Vertex *> &vertices() const { return vertices_; }
  const std::vector<std::shared_ptr<Element2DView>> &elements() const {
    return elements_;
  }

  const std::vector<std::pair<size_t, Element2DView *>> &history() const {
    return history_;
  }
  const datastructures::MultiTreeView<Element2DView> &element_view() const {
    return element_view_;
  }

 protected:
  datastructures::MultiTreeView<Element2DView> element_view_;
  std::vector<Vertex *> vertices_;
  std::vector<std::shared_ptr<Element2DView>> elements_;
  std::vector<std::pair<size_t, Element2DView *>> history_;

  // A convenient helper function for the constructor.
  Vertex *ToVertex(Vertex *v) { return v; }
  Vertex *ToVertex(HierarchicalBasisFn *phi) { return phi->vertex(); }
  template <typename I>
  std::vector<Vertex *> Transform(const std::shared_ptr<I> &root) {
    assert(root->is_root());
    std::vector<Vertex *> result;
    auto nodes = root->Bfs();
    result.reserve(nodes.size());
    for (const auto nv : nodes) result.emplace_back(ToVertex(nv->node()));
    return result;
  }
};
}  // namespace space
