#include "triangulation.hpp"

#include "basis.hpp"

namespace space {

HierarchicalBasisFn *Vertex::RefineHierarchicalBasisFn() {
  // This creates HierarhicalBasisFunctions functions on this vertex.
  if (!phi_) {
    std::vector<HierarchicalBasisFn *> phi_parents;
    for (auto vertex_parent : parents_)
      phi_parents.emplace_back(vertex_parent->RefineHierarchicalBasisFn());
    assert(phi_parents.size());
    phi_ = phi_parents[0]->make_child(/* parents */ phi_parents, this);
    for (int i = 1; i < phi_parents.size(); ++i)
      phi_parents[i]->children_.emplace_back(phi_);
    assert(phi_);
  }

  return phi_;
}

std::array<Vertex *, 2> Element2D::edge(int i) const {
  assert(0 <= i && i <= 2);
  return {{vertices_[(i + 1) % 3], vertices_[(i + 2) % 3]}};
}

std::array<Vertex *, 2> Element2D::reversed_edge(int i) const {
  assert(0 <= i && i <= 2);
  return {{vertices_[(i + 2) % 3], vertices_[(i + 1) % 3]}};
}

bool Element2D::Refine() {
  if (!is_full()) {
    auto nbr = neighbours[0];
    if (!nbr) {  // Refinement edge of `elem` is on domain boundary
      Bisect();
    } else if (nbr->edge(0) != reversed_edge(0)) {
      nbr->Refine();
      return Refine();
    } else {
      BisectWithNbr();
    }
    return true;
  }
  return false;
}

Vertex *Element2D::CreateNewVertex(Element2D *nbr) {
  assert(is_leaf());
  std::vector<Vertex *> vertex_parents{newest_vertex()};
  if (nbr) {
    assert(nbr->edge(0) == reversed_edge(0));
    vertex_parents.emplace_back(nbr->newest_vertex());
  }

  std::array<Vertex *, 2> godparents{{vertices_[1], vertices_[2]}};
  auto new_vertex = vertex_parents[0]->make_child(
      /* parents */ vertex_parents,
      /* x */ (godparents[0]->x + godparents[1]->x) / 2,
      /* y */ (godparents[0]->y + godparents[1]->y) / 2,
      /* on_domain_boundary */ nbr == nullptr);
  if (vertex_parents.size() == 2)
    vertex_parents[1]->children_.emplace_back(new_vertex);
  return new_vertex;
}

std::array<Element2D *, 2> Element2D::Bisect(Vertex *new_vertex) {
  assert(is_leaf());
  if (!new_vertex) {
    new_vertex = CreateNewVertex();
  }
  auto child1 =
      make_child(/* parent */ this, /* vertices */ std::array<Vertex *, 3>{
                     {new_vertex, vertices_[0], vertices_[1]}});
  auto child2 =
      make_child(/* parent */ this, /* vertices */ std::array<Vertex *, 3>{
                     {new_vertex, vertices_[2], vertices_[0]}});
  child1->neighbours = {{neighbours[2], nullptr, child2}};
  child2->neighbours = {{neighbours[1], child1, nullptr}};

  assert(child1->edge(2) == child2->reversed_edge(1));
  new_vertex->patch.push_back(child1);
  new_vertex->patch.push_back(child2);

  if (neighbours[2]) {
    for (int i = 0; i < 3; ++i) {
      if (neighbours[2]->neighbours[i] == this) {
        neighbours[2]->neighbours[i] = child1;
      }
    }
  }
  if (neighbours[1]) {
    for (int i = 0; i < 3; ++i) {
      if (neighbours[1]->neighbours[i] == this) {
        neighbours[1]->neighbours[i] = child2;
      }
    }
  }
  return {{child1, child2}};
}

void Element2D::BisectWithNbr() {
  auto nbr = neighbours[0];
  assert(edge(0) == nbr->reversed_edge(0));

  auto new_vertex = CreateNewVertex(nbr);
  auto children0 = Bisect(new_vertex);
  auto children1 = nbr->Bisect(new_vertex);

  children0[0]->neighbours[1] = children1[1];
  children0[1]->neighbours[2] = children1[0];
  children1[0]->neighbours[1] = children0[1];
  children1[1]->neighbours[2] = children0[0];
}
};  // namespace space
