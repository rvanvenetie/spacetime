#include "hierarchical_basis.hpp"

#include <cmath>
#include <iostream>
#include <memory>
namespace Time {
HierarchicalWaveletFn::HierarchicalWaveletFn(
    Deque<HierarchicalWaveletFn> *container,
    const SmallVector<
        ContLinearScalingFn *,
        datastructures::NodeTrait<ContLinearScalingFn>::N_children>
        &mother_scalings)
    : WaveletFn<HierarchicalWaveletFn>(container) {
  assert(mother_scalings.size() == 2);
  for (size_t i = 0; i < 2; ++i) {
    make_child(
        /* parents */ std::vector{this},
        /* index */ i,
        /* single_scale */
        std::vector{std::pair{mother_scalings[i], 1.0}});
  }

  // Register the metaroot inside the metaroot of element1d.
  mother_scalings[0]->support()[0]->parent()->psi_hierarch_ = this;
}

bool HierarchicalWaveletFn::Refine() {
  if (is_full()) return false;
  assert(children_.empty());

  if (level_ == 0) {
    // Find all wavelets on level 0 -- copy of scaling functions.
    auto parents = this->parents()[0]->children();
    assert(parents.size() == 2 && parents[0]->index_ == 0 &&
           parents[1]->index_ == 1);

    // Find the associated single scale functions
    std::vector<ContLinearScalingFn *> mother_scalings{
        parents[0]->single_scale_[0].first, parents[1]->single_scale_[0].first};

    // Refine the mother scaling function(s).
    mother_scalings[0]->RefineMiddle();
    mother_scalings[0]->RefineRight();
    mother_scalings[1]->RefineMiddle();

    assert(mother_scalings[0]->child_middle_);
    assert(mother_scalings[0]->child_right_);
    assert(mother_scalings[1]->child_middle_);
    assert(parents[0]);
    assert(parents[1]);

    // Create the mother wavelet.
    auto child = parents[0]->make_child(
        /* parents */ std::vector{parents[0], parents[1]},
        /* index */ 0,
        /* single_scale */
        std::vector{std::pair{mother_scalings[0]->child_right_, 1.0}});
    parents[1]->children_ = {child};
  } else {
    assert(level_ > 0);
    auto phi = single_scale_[0].first;

    // Refine the necessary scaling functions.
    assert(phi->parents()[0]->child_middle_);
    assert(phi->parents()[1]->child_middle_);
    phi->RefineMiddle();
    phi->parents()[0]->child_middle_->RefineMiddle();
    phi->parents()[1]->child_middle_->RefineMiddle();
    phi->RefineLeft();
    phi->RefineRight();

    // Create the two children
    make_child(
        /* parents */ std::vector{this},
        /* index */ 2 * index_,
        /* single_scale */
        std::vector{std::pair{phi->child_left_, 1.0}});

    make_child(
        /* parents */ std::vector{this},
        /* index */ 2 * index_ + 1,
        /* single_scale */
        std::vector{std::pair{phi->child_right_, 1.0}});
  }
  return true;
}

bool HierarchicalWaveletFn::is_full() const {
  if (level_ == 0)
    return children_.size() == 1;
  else
    return children_.size() == 2;
}

}  // namespace Time
