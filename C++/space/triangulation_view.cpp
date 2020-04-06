#include "triangulation_view.hpp"

namespace space {

using datastructures::TreeVector;
using datastructures::TreeView;

TriangulationView::TriangulationView(std::vector<Vertex *> &&vertices)
    : vertices_(std::move(vertices)),
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

  // For every new vertex introduced, we store the elements touching
  // the refinement edge.
  history_.resize(vertices_.size());

  // Create and fill in the history object.
  elements_ = element_view_.Bfs();
  history_.reserve(elements_.size());
  for (auto elem : elements_) {
    if (elem->children(0).size() == 0) continue;
    assert(elem->children(0).size() == 2);
    // Get the index of the created vertex by checking a child.
    size_t newest_vertex = elem->children(0)[0]->NewestVertex();
    auto &vertex = vertices_.at(newest_vertex);
    assert(vertex->level() == elem->level() + 1);
    auto &hist = history_.at(newest_vertex);
    assert(hist.size() < 2);
    hist.push_back(elem);
  }

  // Unset the data stored in the vertices.
  for (auto &nv : vertices_) nv->reset_data();
}

}  // namespace space
