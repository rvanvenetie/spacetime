#include "basis.hpp"
namespace Time {

// Initialize static variable;
datastructures::Tree<Element1D> elem_tree;
Element1D* mother_element{elem_tree.meta_root->children()[0].get()};

bool Element1D::Refine() {
  if (is_full()) return false;
  children_.push_back(std::make_shared<Element1D>(this, true));
  children_.push_back(std::make_shared<Element1D>(this, false));
  return true;
}

std::pair<double, double> Element1D::Interval() const {
  assert(!is_metaroot());
  double h = 1.0 / std::pow(2, level_);
  return {h * index_, h * (index_ + 1)};
}

}  // namespace Time
