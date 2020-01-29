#include "triangulation_view.hpp"

namespace space {

using datastructures::TreeVector;
using datastructures::TreeView;

TriangulationView::TriangulationView(std::vector<Vertex *> vertices)
    : vertices_(vertices), element_view_(vertices[0]->patch[0]->parents()[0]) {
  // First, we store a reference to this object in the underlying tree.
  std::vector<size_t> indices(vertices_.size());
  for (size_t i = 0; i < vertices_.size(); ++i) {
    indices[i] = i;
    vertices_[i]->set_data(&indices[i]);
  }

  // Now create the associated element tree
  element_view_.DeepRefine(
      /* call_filter */
      [](auto &&node) { return node->newest_vertex()->has_data(); },
      /* call_postprocess */
      [](auto nv) {
        if (nv->is_root()) return;
        for (size_t i = 0; i < 3; ++i)
          nv->vertices_view_idx_[i] =
              *nv->node()->vertices()[i]->template data<size_t>();
      });

  // Create a history object -- used somehow.
  elements_ = element_view_.Bfs();
  history_.reserve(elements_.size());
  for (auto &&elem : elements_) {
    auto &vertex = vertices_[elem->NewestVertex()];
    if (elem->level() <= 0 || vertex->marked()) continue;
    vertex->set_marked(true);
    assert(!elem->parents().empty());
    history_.emplace_back(elem->NewestVertex(), elem->parents()[0]);
  }

  // Unset all the data!
  for (auto &nv : vertices_) {
    nv->set_marked(false);
    nv->reset_data();
  }
}

// Create some convenient constructors.
std::vector<Vertex *> transform(const TreeView<Vertex> &tree) {
  std::vector<Vertex *> result;
  auto nodes = tree.Bfs();
  result.reserve(nodes.size());
  for (const auto nv : nodes) result.emplace_back(nv->node());
  return result;
}
std::vector<Vertex *> transform(const TreeVector<HierarchicalBasisFn> &tree) {
  std::vector<Vertex *> result;
  auto nodes = tree.Bfs();
  result.reserve(nodes.size());
  for (const auto nv : nodes) result.emplace_back(nv->node()->vertex());
  return result;
}

TriangulationView::TriangulationView(const TreeView<Vertex> &vertex_view)
    : TriangulationView(transform(vertex_view)) {}
TriangulationView::TriangulationView(
    const TreeVector<HierarchicalBasisFn> &basis_vector)
    : TriangulationView(transform(basis_vector)) {}

}  // namespace space
