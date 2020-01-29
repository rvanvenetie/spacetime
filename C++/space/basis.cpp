#include "basis.hpp"

namespace space {

bool HierarchicalBasisFn::Refine() {
  if (is_full()) return false;
  for (auto &elem : vertex_->patch) {
    elem->Refine();
    assert(elem->children().size() == 2);
    elem->children()[0]->newest_vertex()->RefineHierarchicalBasisFn();
  }
  assert(is_full());
  return true;
}

bool HierarchicalBasisFn::is_full() const {
  if (is_metaroot()) return vertex_->children().size() == children().size();
  for (auto &elem : vertex_->patch)
    if (!elem->is_full()) return false;
  return true;
}

double HierarchicalBasisFn::Eval(double x, double y) const {
  for (auto elem : support()) {
    auto bary = elem->BarycentricCoordinates(x, y);

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

Eigen::Vector2d HierarchicalBasisFn::EvalGrad(double x, double y) const {
  for (auto elem : support()) {
    auto bary = elem->BarycentricCoordinates(x, y);

    // Check if the point is contained inside this element.
    if ((bary.array() >= 0).all()) {
      // Find which barycentric coordinate corresponds to this hat fn.
      for (int i = 0; i < 3; ++i)
        if (elem->vertices()[i] == vertex()) {
          auto edge = elem->edge(i);
          Eigen::Vector2d normal{edge[1], -edge[0]};
          normal /= 2.0 * elem->area();
          return normal;
        }
      assert(false);
    }
  }
  return 0;
}

}  // namespace space
