#include "three_point_basis.hpp"

#include <cmath>
#include <iostream>
#include <memory>
namespace Time {

// Initialize static variables.
datastructures::Tree<ContLinearScalingFn> cont_lin_tree;
datastructures::Tree<ThreePointWaveletFn> three_point_tree;

// Metaroot constructor.
ContLinearScalingFn::ContLinearScalingFn() : ScalingFn<ContLinearScalingFn>() {
  auto scaling_left = std::make_shared<ContLinearScalingFn>(
      /* parents */ std::vector{this},
      /* index */ 0,
      /* support */ std::vector{mother_element});
  auto scaling_right = std::make_shared<ContLinearScalingFn>(
      /* parents */ std::vector{this},
      /* index */ 1,
      /* support */ std::vector{mother_element});

  scaling_left->nbr_right_ = scaling_right.get();
  scaling_right->nbr_left_ = scaling_left.get();
  children_.push_back(scaling_left);
  children_.push_back(scaling_right);
}

double ContLinearScalingFn::EvalMother(double t, bool deriv) {
  int left_mask = (-1 < t) && (t <= 0);
  int right_mask = (0 < t) && (t < 1);

  if (deriv)
    return left_mask * 1 + right_mask * -1;
  else
    return left_mask * (1 + t) + right_mask * (1 - t);
}

bool ContLinearScalingFn::RefineMiddle() {
  if (child_middle_) return false;
  auto [l, n] = labda();
  for (auto elem : support_) elem->Refine();

  std::vector<Element1D *> child_support;
  if (n > 0) child_support.push_back(support_[0]->children()[1].get());
  if (n < pow(2, l))
    child_support.push_back(support_.back()->children()[0].get());

  // Create child, and add accordingly.
  auto child = std::make_shared<ContLinearScalingFn>(
      /* parents */ std::vector{this},
      /* index */ 2 * n,
      /* support */ child_support);

  child_middle_ = child.get();
  children_.push_back(child);

  if (child_left_) {
    child_left_->nbr_right_ = child.get();
    child->nbr_left_ = child_left_;
  }
  if (child_right_) {
    child_right_->nbr_left_ = child.get();
    child->nbr_right_ = child_right_;
  }

  return true;
}

bool ContLinearScalingFn::RefineLeft() {
  assert(nbr_left_);
  if (child_left_) return false;
  support_[0]->Refine();
  auto [l, n] = labda();

  // Create child, and add accordingly.
  auto elems = support_[0]->children();
  auto child = std::make_shared<ContLinearScalingFn>(
      /* parents */ std::vector{nbr_left_, this},
      /* index */ (2 * n - 1),
      /* support */ std::vector{elems[0].get(), elems[1].get()});

  // Add this child to ourself.
  child_left_ = child.get();
  children_.push_back(child);

  // Add this child to our left neighbour.
  nbr_left_->child_right_ = child.get();
  nbr_left_->children_.push_back(child);

  // Update neighbours of children.
  if (nbr_left_->child_middle_) {
    nbr_left_->child_middle_->nbr_right_ = child.get();
    child->nbr_left_ = nbr_left_->child_middle_;
  }
  if (child_middle_) {
    child_middle_->nbr_left_ = child.get();
    child->nbr_right_ = child_middle_;
  }

  return true;
}

ThreePointWaveletFn::ThreePointWaveletFn() : WaveletFn<ThreePointWaveletFn>() {
  auto mother_scalings = cont_lin_tree.meta_root->children();
  assert(mother_scalings.size() == 2);
  for (size_t i = 0; i < 2; ++i) {
    children_.push_back(std::make_shared<ThreePointWaveletFn>(
        /* parents */ std::vector{this},
        /* index */ i,
        /* support */ std::vector{std::pair{mother_scalings[i].get(), 1.0}}));
  }
}

bool ThreePointWaveletFn::Refine() {
  if (is_full()) return true;
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
    double sq2 = sqrt(2);
    auto child = std::make_shared<ThreePointWaveletFn>(
        /* parents */ std::vector{parents[0].get(), parents[1].get()},
        /* index */ 0,
        /* single_scale */
        std::vector{std::pair{mother_scalings[0]->child_middle_, -sq2},
                    std::pair{mother_scalings[0]->child_right_, sq2},
                    std::pair{mother_scalings[1]->child_middle_, -sq2}});
    for (const auto parent : parents) parent->children_ = {child};
  } else {
    assert(level_ > 0);
    auto phi_left = single_scale_[0].first;
    auto phi_middle = single_scale_[1].first;
    auto phi_right = single_scale_[2].first;

    // Refine the necessary scaling functions.
    phi_left->RefineMiddle();
    phi_middle->RefineLeft();
    phi_middle->RefineMiddle();
    phi_middle->RefineRight();
    phi_right->RefineMiddle();

    // First refine the left part.
    auto [l, n] = labda();
    double scaling = pow(2, (l + 1) / 2.0);
    std::vector<ContLinearScalingFn *> phi_children{phi_left->child_middle_,
                                                    phi_middle->child_left_,
                                                    phi_middle->child_middle_};

    if (n == 0) {
      children_.push_back(std::make_shared<ThreePointWaveletFn>(
          /* parents */ std::vector{this},
          /* index */ 2 * index_,
          /* single_scale */
          std::vector{std::pair{phi_children[0], -scaling},
                      std::pair{phi_children[1], scaling},
                      std::pair{phi_children[2], -0.5 * scaling}}));
    } else {
      children_.push_back(std::make_shared<ThreePointWaveletFn>(
          /* parents */ std::vector{this},
          /* index */ 2 * index_,
          /* single_scale */
          std::vector{std::pair{phi_children[0], -0.5 * scaling},
                      std::pair{phi_children[1], scaling},
                      std::pair{phi_children[2], -0.5 * scaling}}));
    }

    // Now refine ther right part
    phi_children = {phi_middle->child_middle_, phi_middle->child_right_,
                    phi_right->child_middle_};

    if (n == pow(2, (l - 1)) - 1) {
      children_.push_back(std::make_shared<ThreePointWaveletFn>(
          /* parents */ std::vector{this},
          /* index */ 2 * index_ + 1,
          /* single_scale */
          std::vector{std::pair{phi_children[0], -0.5 * scaling},
                      std::pair{phi_children[1], scaling},
                      std::pair{phi_children[2], -scaling}}));
    } else {
      children_.push_back(std::make_shared<ThreePointWaveletFn>(
          /* parents */ std::vector{this},
          /* index */ 2 * index_ + 1,
          /* single_scale */
          std::vector{std::pair{phi_children[0], -0.5 * scaling},
                      std::pair{phi_children[1], scaling},
                      std::pair{phi_children[2], -0.5 * scaling}}));
    }
  }
  return true;
}

bool ThreePointWaveletFn::is_full() const {
  if (level_ == 0)
    return children_.size() == 1;
  else
    return children_.size() == 2;
}

}  // namespace Time
