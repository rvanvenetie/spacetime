#pragma once
#include <algorithm>
#include <set>
#include <vector>

#include "triangulation_view.hpp"

namespace space {

class MultigridTriangulationView {
 public:
  void Refine();
  void Coarsen();

  inline bool CanRefine() const { return vi_ < triang_.vertices().size(); }
  inline bool CanCoarsen() const { return vi_ > triang_.InitialVertices(); }
  inline bool ContainsVertex(size_t vertex) const { return vertex < vi_; }

  // Return the patches as currently stored inside this object.
  const std::vector<std::vector<Element2DView *>> &patches() const {
    return patches_;
  }

  // Named constructor for initializing object with coarsest mesh patches.
  static MultigridTriangulationView FromCoarsestTriangulation(
      const TriangulationView &triang);

  // Named constructor for initializing object with finest mesh patches.
  static MultigridTriangulationView FromFinestTriangulation(
      const TriangulationView &triang);

  // Debug function that sorts all patches.
  void Sort() {
    for (auto &patch : patches_) std::sort(patch.begin(), patch.end());
  }

 protected:
  const TriangulationView &triang_;
  std::vector<std::vector<Element2DView *>> patches_;
  int vi_;

  // Protected constructor.
  MultigridTriangulationView(const TriangulationView &triang);

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
