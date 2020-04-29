#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "basis.hpp"
#include "datastructures/multi_tree_vector.hpp"
#include "datastructures/multi_tree_view.hpp"
#include "triangulation.hpp"

namespace space {

class TriangulationViewNew {
 public:
  TriangulationViewNew(std::vector<Vertex *> &&vertices)
      : V(vertices.size()),
        vertices_(std::move(vertices)),
        on_boundary_(V),
        godparents_(V, {0, 0}) {
    assert(V >= 3);

    // First, we mark all vertices.
    std::vector<size_t> indices(V);
    initial_vertices_ = 0;
    for (size_t i = 0; i < V; ++i) {
      auto vtx = vertices_[i];
      on_boundary_[i] = vtx->on_domain_boundary;
      indices[i] = i;
      vertices_[i]->set_data(&indices[i]);

      // Find the first vertex that is not of level 0.
      if (vertices_[i]->level() > 0 && initial_vertices_ == 0)
        initial_vertices_ = i;
      assert((vertices_[i]->level() > 0) == (initial_vertices_ > 0));

      // Store link to the Godparents.
      if (vtx->parent_elements.size()) {
        const auto &parent_vertices = vtx->parent_elements[0]->vertices();
        for (int gp = 1; gp < 3; gp++)
          godparents_[i][gp - 1] =
              *parent_vertices[gp]->template data<size_t>();
      }
    }

    // If we only have initial vertices, set the total.
    if (initial_vertices_ == 0) initial_vertices_ = V;

    // Figure out all the element leaves.
    Element2D *elem_meta_root = vertices_[0]->patch[0]->parents()[0];
    assert(elem_meta_root->is_metaroot());

    std::queue<Element2D *> queue;
    element_leaves_.reserve(V * 2);
    for (auto root : elem_meta_root->children()) queue.emplace(root);
    while (!queue.empty()) {
      auto elem = queue.front();
      queue.pop();
      bool is_leaf = true;
      for (const auto &child : elem->children())
        if (child->newest_vertex()->has_data()) {
          queue.emplace(child);
          is_leaf = false;
        }
      if (is_leaf) {
        std::array<size_t, 3> Vids;
        for (size_t i = 0; i < 3; ++i)
          Vids[i] = *elem->vertices()[i]->template data<size_t>();
        element_leaves_.emplace_back(elem, std::move(Vids));
      }
    }

    // Unset the data stored in the vertices.
    for (auto nv : vertices_) nv->reset_data();
  }

  template <typename Iterable>
  TriangulationViewNew(const Iterable &vertices)
      : TriangulationViewNew(Transform(vertices)) {}
  TriangulationViewNew(const datastructures::TreeView<Vertex> &vertex_view)
      : TriangulationViewNew(vertex_view.Bfs()) {}
  TriangulationViewNew(
      const datastructures::TreeVector<HierarchicalBasisFn> &basis_vector)
      : TriangulationViewNew(basis_vector.Bfs()) {}

  TriangulationViewNew InitialTriangulationView() const {
    return TriangulationViewNew(std::vector<Vertex *>(
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
