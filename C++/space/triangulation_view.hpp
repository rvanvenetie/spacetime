#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "datastructures/multi_tree_view.hpp"
#include "triangulation.hpp"

namespace space {

using VertexView = datastructures::NodeView<Vertex>;

class Element2DView
    : public datastructures::NodeViewBase<Element2DView, Element2D> {
 public:
  using datastructures::NodeViewBase<Element2DView, Element2D>::NodeViewBase;

  VertexView *newest_vertex() {
    assert(vertices_view_[0]);
    return vertices_view_[0];
  }

  std::array<VertexView *, 3> vertices_view_{nullptr};
};

class TriangulationView {
 public:
  // Be sure that this treeview doesn't go out of scope!
  TriangulationView(datastructures::TreeView<Vertex> &vertex_view)
      : vertex_view_(vertex_view),
        element_view_(
            vertex_view.root->node()->children()[0]->patch[0]->parents()[0]) {
    // First, we store a reference to this object in the underlying tree.
    for (auto &nv : vertex_view_.nodes()) {
      nv.node()->set_marked(true);
      nv.node()->set_data(&nv);
    }

    // Now create the associated element tree
    element_view_.DeepRefine(
        /* call_filter */
        [](auto &&node) { return node->newest_vertex()->marked(); },
        /* call_postprocess */
        [](Element2DView *nv) {
          if (nv->node()->is_metaroot()) return;
          for (size_t i = 0; i < 3; ++i)
            nv->vertices_view_[i] =
                nv->node()->vertices()[i]->data<VertexView>();
        });

    // Create a history object -- used somehow.
    history_.reserve(vertex_view_.nodes().size());
    element_view_.Bfs(/* include_metaroot */ true,
                      /* callback */
                      [&](auto elem) {
                        if (elem->level() <= 0) return;
                        auto vertex = elem->newest_vertex();
                        if (vertex->marked()) return;
                        vertex->set_marked(true);
                        assert(!elem->parents().empty());
                        history_.emplace_back(vertex, elem->parents()[0]);
                      },
                      /* return_nodes */ false);

    // Unset all the data!
    for (auto &nv : vertex_view_.nodes()) {
      nv.set_marked(false);
      nv.node()->set_marked(false);
      nv.node()->reset_data();
    }
  }
  datastructures::TreeView<Vertex> &vertex_view_;
  datastructures::MultiTree<Element2DView> element_view_;
  std::vector<std::pair<VertexView *, Element2DView *>> history_;
};
}  // namespace space
