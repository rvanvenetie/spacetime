#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "../datastructures/multi_tree_view.hpp"
#include "triangulation.hpp"

namespace space {

using VertexView = datastructures::NodeView<Vertex>;

class Element2DView
    : public datastructures::NodeViewBase<Element2DView, Element2D> {
 public:
  using datastructures::NodeViewBase<Element2DView, Element2D>::NodeViewBase;

  std::shared_ptr<VertexView> newest_vertex() { return vertices_view_[0]; }

  std::array<std::shared_ptr<VertexView>, 3> vertices_view_{nullptr};
};

class TriangulationView {
 public:
  // Be sure that this treeview doesn't go out of scope!
  TriangulationView(datastructures::TreeView<Vertex> &vertex_view)
      : vertex_view_(vertex_view),
        element_view_(vertex_view.root->node()
                          ->children()[0]
                          ->patch[0]
                          ->parents()[0]
                          ->shared_from_this()) {
    // First, we store a reference to this object in the underlying tree.
    auto vertices = vertex_view_.Bfs();
    for (auto &nv : vertices) {
      nv->node()->set_data(&nv);
    }

    // Now create the associated element tree
    element_view_.DeepRefine(
        /* call_filter */
        [](auto &&node) { return node->newest_vertex()->has_data(); },
        /* call_postprocess */
        [](std::shared_ptr<Element2DView> nv) {
          if (nv->is_root()) return;
          for (size_t i = 0; i < 3; ++i)
            nv->vertices_view_[i] =
                *nv->node()->vertices()[i]->data<std::shared_ptr<VertexView>>();
        });

    // Create a history object -- used somehow.
    auto elements = element_view_.Bfs();
    history_.reserve(elements.size());
    for (auto &&elem : elements) {
      auto &&vertex = elem->newest_vertex();
      if (elem->level() <= 0 || vertex->marked()) continue;
      vertex->set_marked(true);
      assert(!elem->parents().empty());
      history_.emplace_back(vertex.get(), elem->parents()[0]);
    }

    // Unset all the data!
    for (auto &nv : vertices) {
      nv->set_marked(false);
      nv->node()->reset_data();
    }
  }
  datastructures::TreeView<Vertex> &vertex_view_;
  datastructures::MultiTreeView<Element2DView> element_view_;
  std::vector<std::pair<VertexView *, Element2DView *>> history_;
};
}  // namespace space
