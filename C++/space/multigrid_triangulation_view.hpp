#pragma once
#include <algorithm>
#include <set>
#include <vector>

#include "datastructures/multi_tree_vector.hpp"
#include "datastructures/multi_tree_view.hpp"
#include "triangulation.hpp"

namespace space {

class Element2DView
    : public datastructures::MultiNodeViewBase<Element2DView, Element2D> {
 public:
  using datastructures::MultiNodeViewBase<Element2DView,
                                          Element2D>::MultiNodeViewBase;

  size_t NewestVertex() const { return vertices_view_idx_[0]; }
  inline std::array<size_t, 2> RefinementEdge() const {
    return {vertices_view_idx_[1], vertices_view_idx_[2]};
  }

  std::array<size_t, 3> vertices_view_idx_;
};

class MultigridTriangulationView {
 public:
  // Constructor.
  MultigridTriangulationView(const std::vector<Vertex *> &vertices,
                             bool initialize_finest_level = true);

  void Refine();
  void Coarsen();

  inline bool CanRefine() const { return vi_ < V; }
  inline bool CanCoarsen() const { return vi_ > initial_vertices_; }
  inline bool ContainsVertex(size_t vertex) const { return vertex < vi_; }

  // Return the patches as currently stored inside this object.
  const std::vector<std::vector<Element2DView *>> &patches() const {
    return patches_;
  }

  const datastructures::MultiTreeView<Element2DView> &element_view() const {
    return element_view_;
  }

  // Debug function that sorts all patches.
  void Sort() {
    for (auto &patch : patches_) std::sort(patch.begin(), patch.end());
  }

  // Total number of vertices.
  const size_t V;

 protected:
  datastructures::MultiTreeView<Element2DView> element_view_;
  std::vector<StaticVector<Element2DView *, 2>> history_;
  std::vector<std::vector<Element2DView *>> patches_;
  int vi_;
  size_t initial_vertices_;

  // Erase/insert element into the patch.
  inline bool Erase(int v, Element2DView *elem) {
    auto &patch = patches_.at(v);
    for (int e = 0; e < patch.size(); e++)
      if (patch[e] == elem) {
        patch[e] = patch.back();
        patch.resize(patch.size() - 1);
        return true;
      }
    return false;
  }
  inline void Insert(int v, Element2DView *elem) {
    patches_[v].emplace_back(elem);
  }
};

}  // namespace space
