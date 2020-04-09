#include "multigrid_triangulation_view.hpp"

namespace space {

void MultigridTriangulationView::Refine() {
  assert(vi_ < triang_.vertices().size());
  assert(patches_[vi_].empty());

  const auto &hist = triang_.history(vi_);
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
  assert(vi_ > triang_.InitialVertices());
  vi_--;
  assert(!patches_[vi_].empty());

  const auto &hist = triang_.history(vi_);
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
    const TriangulationView &triang)
    : triang_(triang) {
  patches_.resize(triang.vertices().size());
}

MultigridTriangulationView
MultigridTriangulationView::FromCoarsestTriangulation(
    const TriangulationView &triang) {
  MultigridTriangulationView result(triang);
  for (const auto &elem : triang.elements()) {
    if (elem->level()) continue;
    for (int vi : elem->vertices_view_idx_) result.Insert(vi, elem);
  }
  result.vi_ = triang.InitialVertices();
  return result;
}

MultigridTriangulationView MultigridTriangulationView::FromFinestTriangulation(
    const TriangulationView &triang) {
  MultigridTriangulationView result(triang);
  for (const auto &elem : triang.elements()) {
    if (!elem->is_leaf()) continue;
    for (int vi : elem->vertices_view_idx_) result.Insert(vi, elem);
  }
  result.vi_ = triang.vertices().size();
  return result;
}

}  // namespace space
