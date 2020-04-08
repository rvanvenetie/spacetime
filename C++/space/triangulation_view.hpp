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
  std::array<size_t, 2> RefinementEdge() const {
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

  size_t InitialVertices() const { return initial_vertices_; }

  const std::vector<Vertex *> &vertices() const { return vertices_; }
  const std::vector<Element2DView *> &elements() const { return elements_; }
  const StaticVector<Element2DView *, 2> &history(int i) const {
    return history_.at(i);
  }

  const datastructures::MultiTreeView<Element2DView> &element_view() const {
    return element_view_;
  }

 protected:
  datastructures::MultiTreeView<Element2DView> element_view_;
  std::vector<Vertex *> vertices_;
  std::vector<Element2DView *> elements_;
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
