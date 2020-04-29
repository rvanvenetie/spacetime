#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "../datastructures/multi_tree_vector.hpp"
#include "../datastructures/multi_tree_view.hpp"
#include "basis.hpp"
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

class TriangulationView {
 public:
  TriangulationView(std::vector<Vertex *> &&vertices);
  template <typename Iterable>
  TriangulationView(const Iterable &vertices)
      : TriangulationView(Transform(vertices)) {}
  TriangulationView(const datastructures::TreeView<Vertex> &vertex_view)
      : TriangulationView(vertex_view.Bfs()) {}
  TriangulationView(
      const datastructures::TreeVector<HierarchicalBasisFn> &basis_vector)
      : TriangulationView(basis_vector.Bfs()) {}

  TriangulationView InitialTriangulationView() const {
    return TriangulationView(std::vector<Vertex *>(
        vertices_.begin(), vertices_.begin() + InitialVertices()));
  };

  // Total number of vertices.
  const size_t V;

  // Does the given vertex lie on the domain boundary?
  inline bool OnBoundary(size_t v) const {
    return vertices_.at(v)->on_domain_boundary;
  }

  // Number of initial vertices.
  inline size_t InitialVertices() const { return initial_vertices_; }

  // Grandparents
  inline const std::array<size_t, 2> Godparents(size_t vi) const {
    const auto &hist = history(vi);
    assert(!hist.empty());
    return hist[0]->RefinementEdge();
  }

  // Access data members.
  inline const std::vector<Vertex *> &vertices() const { return vertices_; }
  inline const std::vector<Element2DView *> &element_leaves() const {
    return leaves_;
  }
  inline const StaticVector<Element2DView *, 2> &history(int i) const {
    return history_.at(i);
  }
  inline const auto &element_view() const { return element_view_; }

 protected:
  datastructures::MultiTreeView<Element2DView> element_view_;
  std::vector<Vertex *> vertices_;
  std::vector<Element2DView *> leaves_;
  std::vector<StaticVector<Element2DView *, 2>> history_;
  size_t initial_vertices_;

  // A convenient helper function for the constructor.
  inline static Vertex *ToVertex(Vertex *v) { return v; }
  inline static Vertex *ToVertex(HierarchicalBasisFn *phi) {
    return phi->vertex();
  }
  template <typename Iterable>
  std::vector<Vertex *> Transform(const Iterable &nodes) {
    std::vector<Vertex *> result;
    assert(nodes.size());
    result.reserve(nodes.size());
    for (const auto nv : nodes) result.emplace_back(ToVertex(nv->node()));
    return result;
  }
};
}  // namespace space
