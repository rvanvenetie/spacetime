#pragma once
#include <set>
#include <vector>

#include "triangulation_view.hpp"

namespace space {

class MultilevelPatches {
 public:
  void Refine();
  void Coarsen();

  // What vertex is the last vertex that was added?
  int CurrentVertex() const { return vi_; }

  bool CanRefine() const { return vi_ < triang_.vertices().size(); }
  bool CanCoarsen() const { return vi_ > triang_.InitialVertices(); }

  // Return the patches as currently stored inside this object.
  const std::vector<std::set<Element2DView *>> &patches() const {
    return patches_;
  }

  // Named constructor for initializing object with coarsest mesh patches.
  static MultilevelPatches FromCoarsestTriangulation(
      const TriangulationView &triang);

  // Named constructor for initializing object with finest mesh patches.
  static MultilevelPatches FromFinestTriangulation(
      const TriangulationView &triang);

 protected:
  const TriangulationView &triang_;
  std::vector<std::set<Element2DView *>> patches_;
  int vi_;

  // Protected constructor.
  MultilevelPatches(const TriangulationView &triang);
};

}  // namespace space
