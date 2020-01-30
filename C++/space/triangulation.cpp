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

Eigen::Vector3d Element2D::BarycentricCoordinates(double x, double y) const {
  // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
  Eigen::Vector2d p, a, b, c;
  p << x, y;
  a << vertices_[0]->x, vertices_[0]->y;
  b << vertices_[1]->x, vertices_[1]->y;
  c << vertices_[2]->x, vertices_[2]->y;

  auto v0 = b - a, v1 = c - a, v2 = p - a;

  auto d00 = v0.dot(v0);
  auto d01 = v0.dot(v1);
  auto d11 = v1.dot(v1);
  auto d20 = v2.dot(v0);
  auto d21 = v2.dot(v1);

  double denom = (d00 * d11 - d01 * d01);
  double v = (d11 * d20 - d01 * d21) / denom;
  double w = (d00 * d21 - d01 * d20) / denom;

  Eigen::Vector3d bary;
  bary << 1 - v - w, v, w;
  return bary;
}

std::pair<double, double> Element2D::GlobalCoordinates(double bary2,
                                                       double bary3) const {
  assert(0 <= bary2 && bary2 <= 1 && 0 <= bary3 && bary3 <= 1);
  const auto &V = vertices_;
  return {(V[1]->x - V[0]->x) * bary2 + (V[2]->x - V[0]->x) * bary3 + V[0]->x,
          (V[1]->y - V[0]->y) * bary2 + (V[2]->y - V[0]->y) * bary3 + V[0]->y};
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
