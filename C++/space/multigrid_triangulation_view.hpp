#pragma once
#include <set>
#include <vector>

#include "triangulation_view.hpp"

namespace space {

class MultigridTriangulationView {
 public:
  void Refine();
  void Coarsen();

  // Index of the vertex that is to be added by a refine.
  int CurrentVertex() const { return vi_; }

  bool CanRefine() const { return vi_ < triang_.vertices().size(); }
  bool CanCoarsen() const { return vi_ > triang_.InitialVertices(); }

  bool ContainsVertex(size_t vertex) const { return vertex < vi_; }

  // Return the patches as currently stored inside this object.
  const std::vector<std::set<Element2DView *>> &patches() const {
    return patches_;
  }

  // Named constructor for initializing object with coarsest mesh patches.
  static MultigridTriangulationView FromCoarsestTriangulation(
      const TriangulationView &triang);

  // Named constructor for initializing object with finest mesh patches.
  static MultigridTriangulationView FromFinestTriangulation(
      const TriangulationView &triang);

 protected:
  const TriangulationView &triang_;
  std::vector<std::set<Element2DView *>> patches_;
  int vi_;

  // Protected constructor.
  MultigridTriangulationView(const TriangulationView &triang);
};

}  // namespace space
