#include "basis.hpp"

#include "hierarchical_basis.hpp"
#include "orthonormal_basis.hpp"
#include "three_point_basis.hpp"

namespace Time {

bool Element1D::Refine() {
  if (is_full()) return false;
  make_child(/* parent */ this, /* left_child */ true);
  make_child(/* parent */ this, /* left_child */ false);
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

const std::array<OrthonormalWaveletFn *, 2> &Element1D::RefinePsiOrthonormal() {
  if (!psi_ortho_[0] || !psi_ortho_[1]) {
    assert(level() > 0);
    const auto &psi_parent = parent()->RefinePsiOrthonormal();
    psi_parent[0]->Refine();
    assert(psi_ortho_[0] && psi_ortho_[1]);
  }
  return psi_ortho_;
}

HierarchicalWaveletFn *Element1D::RefinePsiHierarchical() {
  assert(this->level_ != 0);

  if (!psi_hierarch_) {
    // There are two HierarchicalWavelets on level 0, so need to do
    // a trick.
    if (this->level_ == 1) {
      assert(parent()->parent()->psi_hierarch_);
      parent()->parent()->psi_hierarch_->children().at(0)->Refine();
    } else {
      parent()->RefinePsiHierarchical()->Refine();
    }
  }

  assert(psi_hierarch_);
  return psi_hierarch_;
}

std::pair<double, double> Element1D::Interval() const {
  assert(!is_metaroot());
  double h = 1.0 / (1LL << level_);
  return {h * index_, h * (index_ + 1)};
}

double Element1D::GlobalCoordinates(double bary2) const {
  assert(0 <= bary2 && bary2 <= 1);
  auto [a, b] = Interval();
  return a + bary2 * (b - a);
}

}  // namespace Time
