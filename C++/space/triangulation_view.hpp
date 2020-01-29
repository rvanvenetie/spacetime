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
  TriangulationView(const datastructures::TreeView<Vertex> &vertex_view);
  TriangulationView(
      const datastructures::TreeVector<HierarchicalBasisFn> &basis_vector);

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
};
}  // namespace space
