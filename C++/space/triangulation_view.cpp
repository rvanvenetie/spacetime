#include "triangulation_view.hpp"

namespace space {

TriangulationView::TriangulationView(std::vector<Vertex *> &&vertices)
    : V(vertices.size()),
      vertices_(std::move(vertices)),
      on_boundary_(V),
      godparents_(V, {0, 0}) {
  assert(V >= 3);

  // First, we mark all vertices.
  std::vector<uint> indices(V);
  initial_vertices_ = 0;
  for (uint i = 0; i < V; ++i) {
    auto vtx = vertices_[i];
    on_boundary_[i] = vtx->on_domain_boundary;
    indices[i] = i;
    vertices_[i]->set_data(&indices[i]);

    // Find the first vertex that is not of level 0.
    if (initial_vertices_ == 0 && vertices_[i]->level() > 0)
      initial_vertices_ = i;

    // Store link to the Godparents.
    if (vtx->godparents.size())
      for (int gp = 0; gp < 2; gp++)
        godparents_[i][gp] = *vtx->godparents[gp]->template data<uint>();
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
      std::array<uint, 3> Vids;
      for (uint i = 0; i < 3; ++i)
        Vids[i] = *elem->vertices()[i]->template data<uint>();
      element_leaves_.emplace_back(elem, std::move(Vids));
    }
  }

  // Unset the data stored in the vertices.
  for (auto nv : vertices_) nv->reset_data();
}

}  // namespace space
