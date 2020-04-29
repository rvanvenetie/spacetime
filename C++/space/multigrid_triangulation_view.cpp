#include "multigrid_triangulation_view.hpp"

namespace space {

void MultigridTriangulationView::Refine() {
  assert(vi_ < V);
  assert(patches_[vi_].empty());

  const auto &hist = history_[vi_];
  assert(!hist.empty());

  // We must remove the elements from the existing patches.
  for (auto elem : hist)
    for (auto v : elem->vertices_view_idx_) assert(Erase(v, elem));

  // Now we must update the new elements.
  for (auto elem : hist)
    for (auto child : elem->children(0))
      for (auto v : child->vertices_view_idx_) Insert(v, child);
  vi_++;
}

void MultigridTriangulationView::Coarsen() {
  assert(vi_ > initial_vertices_);
  vi_--;
  assert(!patches_[vi_].empty());

  const auto &hist = history_[vi_];
  assert(!hist.empty());

  // We must remove the elements from the existing patches.
  for (auto elem : hist)
    for (auto child : elem->children(0))
      for (auto v : child->vertices_view_idx_) assert(Erase(v, child));

  // Now we must update the new elements.
  for (auto elem : hist)
    for (auto v : elem->vertices_view_idx_) Insert(v, elem);
}

MultigridTriangulationView::MultigridTriangulationView(
    const std::vector<Vertex *> &vertices, bool initialize_finest_level)
    : V(vertices.size()), element_view_(vertices[0]->patch[0]->parents()[0]) {
  // First, we store a reference to this object in the underlying tree.
  std::vector<size_t> indices(V);
  initial_vertices_ = 0;
  for (size_t i = 0; i < V; ++i) {
    indices[i] = i;
    vertices[i]->set_data(&indices[i]);

    // Find the first vertex that is not of level 0.
    if (vertices[i]->level() > 0 && initial_vertices_ == 0)
      initial_vertices_ = i;
    assert((vertices[i]->level() > 0) == (initial_vertices_ > 0));
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
  for (auto elem : elements) {
    if (elem->children(0).size() == 0) continue;
    assert(elem->children(0).size() == 2);
    // Get the index of the created vertex by checking a child.
    size_t newest_vertex = elem->children(0)[0]->NewestVertex();
    auto &vertex = vertices[newest_vertex];
    assert(vertex->level() == elem->level() + 1);
    auto &hist = history_[newest_vertex];
    assert(hist.size() < 2);
    hist.push_back(elem);
  }

  // Unset the data stored in the vertices.
  for (auto &nv : vertices) nv->reset_data();

  // Reserve size in the patches.
  patches_.resize(V);
  for (auto &patch : patches_) patch.reserve(4);

  if (initialize_finest_level) {
    // Initialize with the triangulation on the finest level.
    for (auto elem : elements) {
      if (!elem->is_leaf()) continue;
      for (int vi : elem->vertices_view_idx_) Insert(vi, elem);
    }
    vi_ = V;
  } else {
    // Initialize with the triangulation on the coarest level.
    for (const auto &elem : element_view_.root()->children(0)) {
      assert(elem->level() == 0);
      for (int vi : elem->vertices_view_idx_) Insert(vi, elem);
    }
    vi_ = initial_vertices_;
  }
}

}  // namespace space
