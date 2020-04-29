#include "triangulation_view.hpp"

namespace space {

using datastructures::TreeVector;
using datastructures::TreeView;

TriangulationView::TriangulationView(std::vector<Vertex *> &&vertices)
    : V(vertices.size()),
      vertices_(std::move(vertices)),
      element_view_(vertices[0]->patch[0]->parents()[0]) {
  assert(V >= 3);
  // First, we store a reference to this object in the underlying tree.
  std::vector<size_t> indices(V);
  initial_vertices_ = 0;
  for (size_t i = 0; i < V; ++i) {
    indices[i] = i;
    vertices_[i]->set_data(&indices[i]);

    // Find the first vertex that is not of level 0.
    if (vertices_[i]->level() > 0 && initial_vertices_ == 0)
      initial_vertices_ = i;
    assert((vertices_[i]->level() > 0) == (initial_vertices_ > 0));
  }

  // If we only have initial vertices, set the total.
  if (initial_vertices_ == 0) initial_vertices_ = V;

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
  history_.resize(V);

  // Create and fill in the history object.
  auto elements = element_view_.Bfs();
  history_.reserve(elements.size());
  leaves_.reserve(elements.size() / 2);
  for (auto elem : elements) {
    if (elem->is_leaf()) leaves_.emplace_back(elem);
    if (elem->children(0).size() == 0) continue;
    assert(elem->children(0).size() == 2);
    // Get the index of the created vertex by checking a child.
    size_t newest_vertex = elem->children(0)[0]->NewestVertex();
    auto &vertex = vertices_[newest_vertex];
    assert(vertex->level() == elem->level() + 1);
    auto &hist = history_[newest_vertex];
    assert(hist.size() < 2);
    hist.push_back(elem);
  }

  // Unset the data stored in the vertices.
  for (auto &nv : vertices_) nv->reset_data();
}

}  // namespace space
