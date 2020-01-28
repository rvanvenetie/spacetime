#pragma once
#include <Eigen/Dense>

#include "../datastructures/tree.hpp"
#include "triangulation.hpp"

namespace space {

class HierarchicalBasisFn : public datastructures::Node<HierarchicalBasisFn> {
 public:
  static constexpr size_t N_parents = 2;
  static constexpr size_t N_children = 4;

  HierarchicalBasisFn(const std::vector<HierarchicalBasisFn *> &parents,
                      Vertex *vertex)
      : Node(parents), vertex_(vertex) {}

  bool Refine() {
    if (is_full()) return false;
    for (auto &elem : vertex_->patch) {
      elem->Refine();
      assert(elem->children().size() == 2);
      elem->children()[0]->newest_vertex()->RefineHierarchicalBasisFn();
    }
    assert(is_full());
    return true;
  }

  bool is_full() const {
    if (is_metaroot()) return vertex_->children().size() == children().size();
    for (auto &elem : vertex_->patch)
      if (!elem->is_full()) return false;
    return true;
  }

  Vertex *vertex() const { return vertex_; }
  const std::vector<Element2D *> &support() const { return vertex_->patch; }
  double eval(double x, double y) const {
    for (auto elem : support()) {
      // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
      Eigen::Vector2d p, a, b, c;
      p << x, y;
      a << elem->vertices()[0]->x, elem->vertices()[0]->y;
      b << elem->vertices()[1]->x, elem->vertices()[1]->y;
      c << elem->vertices()[2]->x, elem->vertices()[2]->y;
      auto v0 = b - a;
      auto v1 = c - a;
      auto v2 = p - a;
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

      // Check if the point is contained inside this element.
      if ((bary.array() >= 0).all()) {
        // Find which barycentric coordinate corresponds to this hat fn.
        for (int i = 0; i < 3; ++i)
          if (elem->vertices()[i] == vertex()) return bary[i];
        assert(false);
      }
    }
    return 0;
  }

 private:
  Vertex *vertex_;

  // Protected constructor for creating a metaroot.
  HierarchicalBasisFn(Vertex *vertex) : Node(), vertex_(vertex) {
    assert(vertex->is_metaroot());
  }

  friend datastructures::Tree<HierarchicalBasisFn>;
  friend InitialTriangulation;
  friend Vertex;
};

}  // namespace space
