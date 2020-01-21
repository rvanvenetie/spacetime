#include "basis.hpp"

#include "haar_basis.hpp"
#include "three_point_basis.hpp"

namespace Time {

// Initialize static variable;
datastructures::Tree<Element1D> elem_tree;
Element1D *mother_element{elem_tree.meta_root->children()[0].get()};

bool Element1D::Refine() {
  if (is_full()) return false;
  children_.push_back(std::make_shared<Element1D>(this, true));
  children_.push_back(std::make_shared<Element1D>(this, false));
  return true;
}

const std::array<ContLinearScalingFn *, 2> &Element1D::RefineContLinear() {
  if (!phi_cont_lin_[0] || !phi_cont_lin_[1]) {
    auto phi_parents = parent()->RefineContLinear();
    // Check whether we are the left child element.
    if (index_ % 2 == 0) {
      phi_parents[0]->RefineMiddle();
      phi_parents[0]->RefineRight();
    } else {
      phi_parents[1]->RefineLeft();
      phi_parents[1]->RefineMiddle();
    }
    assert(phi_cont_lin_[0] && phi_cont_lin_[1]);
  }
  return phi_cont_lin_;
}

std::pair<double, double> Element1D::Interval() const {
  assert(!is_metaroot());
  double h = 1.0 / std::pow(2, level_);
  return {h * index_, h * (index_ + 1)};
}

}  // namespace Time
