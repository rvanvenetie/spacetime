#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "basis.hpp"
#include "datastructures/multi_tree_vector.hpp"
#include "datastructures/multi_tree_view.hpp"
#include "triangulation.hpp"

namespace space {

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
  inline bool OnBoundary(size_t v) const { return on_boundary_.at(v); }

  // Number of initial vertices.
  inline size_t InitialVertices() const { return initial_vertices_; }

  // Grandparents
  inline const std::array<size_t, 2> &Godparents(size_t vi) const {
    return godparents_.at(vi);
  }

  // Access data members.
  inline const std::vector<std::pair<Element2D *, std::array<size_t, 3>>>
      &element_leaves() const {
    return element_leaves_;
  }
  const std::vector<Vertex *> &vertices() const { return vertices_; }

 protected:
  std::vector<Vertex *> vertices_;
  std::vector<bool> on_boundary_;
  std::vector<std::array<size_t, 2>> godparents_;
  std::vector<std::pair<Element2D *, std::array<size_t, 3>>> element_leaves_;
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
