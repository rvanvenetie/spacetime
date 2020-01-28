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
  TriangulationView(const datastructures::TreeView<Vertex> &vertex_view)
      : TriangulationView(transform(vertex_view)) {}
  TriangulationView(
      const datastructures::TreeVector<HierarchicalBasisFn> &basis_vector)
      : TriangulationView(transform(basis_vector)) {}

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

  TriangulationView(std::vector<Vertex *> vertices)
      : vertices_(vertices),
        element_view_(vertices[0]->patch[0]->parents()[0]) {
    // First, we store a reference to this object in the underlying tree.
    std::vector<size_t> indices(vertices_.size());
    for (size_t i = 0; i < vertices_.size(); ++i) {
      indices[i] = i;
      vertices_[i]->set_data(&indices[i]);
    }

    // Now create the associated element tree
    element_view_.DeepRefine(
        /* call_filter */
        [](auto &&node) { return node->newest_vertex()->has_data(); },
        /* call_postprocess */
        [](auto nv) {
          if (nv->is_root()) return;
          for (size_t i = 0; i < 3; ++i)
            nv->vertices_view_idx_[i] =
                *nv->node()->vertices()[i]->template data<size_t>();
        });

    // Create a history object -- used somehow.
    elements_ = element_view_.Bfs();
    history_.reserve(elements_.size());
    for (auto &&elem : elements_) {
      auto &vertex = vertices_[elem->NewestVertex()];
      if (elem->level() <= 0 || vertex->marked()) continue;
      vertex->set_marked(true);
      assert(!elem->parents().empty());
      history_.emplace_back(elem->NewestVertex(), elem->parents()[0]);
    }

    // Unset all the data!
    for (auto &nv : vertices_) {
      nv->set_marked(false);
      nv->reset_data();
    }
  }

  static std::vector<Vertex *> transform(
      const datastructures::TreeView<Vertex> &tree) {
    std::vector<Vertex *> result;
    auto nodes = tree.Bfs();
    result.reserve(nodes.size());
    for (const auto nv : nodes) result.emplace_back(nv->node());
    return result;
  }
  static std::vector<Vertex *> transform(
      const datastructures::TreeVector<HierarchicalBasisFn> &tree) {
    std::vector<Vertex *> result;
    auto nodes = tree.Bfs();
    result.reserve(nodes.size());
    for (const auto nv : nodes) result.emplace_back(nv->node()->vertex());
    return result;
  }
};
}  // namespace space
