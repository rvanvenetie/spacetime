#include "multigrid_triangulation_view.hpp"

namespace space {

inline bool MultigridTriangulationView::Erase(int v, Element2DView *elem) {
  auto &patch = patches_.at(v);
  for (int e = 0; e < patch.size(); e++)
    if (patch[e] == elem) {
      patch[e] = patch.back();
      patch.resize(patch.size() - 1);
      return true;
    }
  return false;
}

inline void MultigridTriangulationView::Insert(int v, Element2DView *elem) {
  patches_[v].emplace_back(elem);
}

void MultigridTriangulationView::Refine() {
  assert(vi_ < V);
  assert(patches_[vi_].empty());

  const auto &hist = history_[vi_];
  assert(!hist.empty());

  // We must remove the elements from the existing patches.
  for (auto elem : hist)
    for (auto v : elem->Vids()) assert(Erase(v, elem));

  // Now we must update the new elements.
  for (auto elem : hist)
    for (auto child : elem->children())
      for (auto v : child->Vids()) Insert(v, child);
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
    for (auto child : elem->children())
      for (auto v : child->Vids()) assert(Erase(v, child));

  // Now we must update the new elements.
  for (auto elem : hist)
    for (auto v : elem->Vids()) Insert(v, elem);
}

MultigridTriangulationView::MultigridTriangulationView(
    const std::vector<Vertex *> &vertices, bool initialize_finest_level)
    : V(vertices.size()) {
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

  // Figure out all the element leaves.
  Element2D *elem_meta_root = vertices[0]->patch[0]->parents()[0];
  assert(elem_meta_root->is_metaroot());

  // Now create the associated element tree
  std::queue<Element2DView *> queue;

  // Create the roots and push them on the root queue.
  for (auto root : elem_meta_root->children()) {
    elements_.emplace_back(root);
    queue.emplace(&elements_.back());
  }

  while (!queue.empty()) {
    auto elem_view = queue.front();
    queue.pop();
    for (auto child : elem_view->node()->children())
      if (child->newest_vertex()->has_data()) {
        elements_.emplace_back(child);
        Element2DView *child_view = &elements_.back();
        elem_view->children().emplace_back(child_view);
        queue.emplace(child_view);
      }
  }

  // For every new vertex introduced, we store the elements touching
  // the refinement edge.
  history_.resize(V);

  // Create and fill in the history object.
  for (auto &elem : elements_) {
    if (elem.children().size() == 0) continue;
    assert(elem.children().size() == 2);
    // Get the index of the created vertex by checking a child.
    size_t newest_vertex = elem.children()[0]->NewestVertex();
    auto &vertex = vertices[newest_vertex];
    assert(vertex->level() == elem.level() + 1);
    auto &hist = history_[newest_vertex];
    assert(hist.size() < 2);
    hist.push_back(&elem);
  }

  // Unset the data stored in the vertices.
  for (auto nv : vertices) nv->reset_data();

  // Reserve size in the patches.
  patches_.resize(V);
  for (auto &patch : patches_) patch.reserve(4);

  if (initialize_finest_level) {
    // Initialize with the triangulation on the finest level.
    for (auto &elem : elements_) {
      if (!elem.is_leaf()) continue;
      for (int vi : elem.Vids()) Insert(vi, &elem);
    }
    vi_ = V;
  } else {
    // Initialize with the triangulation on the coarsest level.
    for (auto &elem : elements_) {
      if (elem.level()) break;
      for (int vi : elem.Vids()) Insert(vi, &elem);
    }
    vi_ = initial_vertices_;
  }
}

}  // namespace space
