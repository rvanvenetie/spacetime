#pragma once
#include <algorithm>
#include <set>
#include <vector>

#include "triangulation.hpp"

namespace space {

class Element2DView {
 public:
  Element2DView(Element2D *node) : node_(node) {
    for (uint i = 0; i < 3; ++i)
      vertices_view_idx_[i] = *node->vertices()[i]->template data<uint>();
  }

  // The element this view represents.
  Element2D *node() const { return node_; }
  int level() const { return node_->level(); }
  bool is_leaf() const { return children_.empty(); }

  StaticVector<Element2DView *, 2> &children() { return children_; }
  const StaticVector<Element2DView *, 2> &children() const { return children_; }

  inline const std::array<uint, 3> &Vids() const { return vertices_view_idx_; }
  uint NewestVertex() const { return vertices_view_idx_[0]; }
  inline std::array<uint, 2> RefinementEdge() const {
    return {vertices_view_idx_[1], vertices_view_idx_[2]};
  }

 protected:
  Element2D *node_;
  StaticVector<Element2DView *, 2> children_;
  std::array<uint, 3> vertices_view_idx_;
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
  inline bool ContainsVertex(uint vertex) const { return vertex < vi_; }

  // Return the patches as currently stored inside this object.
  const std::vector<std::vector<Element2DView *>> &patches() const {
    return patches_;
  }

  const std::deque<Element2DView> &elements() const { return elements_; }

  // Total number of vertices.
  const uint V;

 protected:
  std::deque<Element2DView> elements_;

  std::vector<StaticVector<Element2DView *, 2>> history_;
  std::vector<std::vector<Element2DView *>> patches_;
  int vi_;
  uint initial_vertices_;

  // Erase/insert element into the patch.
  inline bool Erase(int v, Element2DView *elem);
  inline void Insert(int v, Element2DView *elem);
};

}  // namespace space
