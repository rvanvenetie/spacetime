#include "three_point_basis.hpp"

#include <iostream>
#include <memory>
namespace Time {

// Initialize static variables.
datastructures::Tree<ContLinearScalingFn> disc_cons_tree;
datastructures::Tree<ThreePointWaveletFn> haar_tree;

// Metaroot constructor.
ContLinearScalingFn::ContLinearScalingFn() : ScalingFn<ContLinearScalingFn>() {
  children_.push_back(
      std::make_shared<ContLinearScalingFn>({this}, 0, {mother_element}));
  children_.push_back(
      std::make_shared<ContLinearScalingFn>({this}, 1, {mother_element}));
}

double ContLinearScalingFn::EvalMother(double t, bool deriv) {
  int left_mask = (-1 < t) && (t <= 0);
  int right_mask = (0 < t) && (t < 1);

  if (deriv)
    return left_mask * 1 + right_mask * -1;
  else
    return left_mask * (1 + t) + right_mask * (1 - t);
}

bool ContLinearScalingFn::Refine() {
  if (is_full()) return false;
  assert(children_.empty());

  support_[0]->Refine();
  auto [l, n] = labda();

  // clang-format off
  children_.push_back(std::make_shared<ContLinearScalingFn>(
      /* parent */ this,
      /* index */ 2 * n,
      /* support */ support_[0]->children()[0].get()));
  children_.push_back(std::make_shared<ContLinearScalingFn>(
      /* parent */ this,
      /* index */ 2 * n + 1,
      /* support */ support_[0]->children()[1].get()));
  // clang-format on

  return true;
}

ThreePointWaveletFn::ThreePointWaveletFn() : WaveletFn<ThreePointWaveletFn>() {
  auto mother_scaling = disc_cons_tree.meta_root->children()[0].get();
  children_.push_back(std::make_shared<ThreePointWaveletFn>(
      this, 0, std::vector{std::pair{mother_scaling, 1.0}}));
}

bool ThreePointWaveletFn::Refine() {
  if (is_full()) return true;
  assert(children_.empty());

  if (level_ == 0) {
    auto mother_scaling = single_scale_[0].first;
    mother_scaling->Refine();
    auto mother_scaling_children = mother_scaling->children();

    children_.push_back(std::make_shared<ThreePointWaveletFn>(
        /* parent */ this,
        /* index */ 0,
        /* single_scale */
        std::vector{std::pair{mother_scaling_children[0].get(), 1.0},
                    std::pair{mother_scaling_children[1].get(), -1.0}}));
  } else {
    assert(level_ > 0);
    auto phi_left = single_scale_[0].first;
    auto phi_right = single_scale_[1].first;
    phi_left->Refine();
    phi_right->Refine();

    children_.push_back(std::make_shared<ThreePointWaveletFn>(
        /* parent */ this,
        /* index */ 2 * index_,
        /* single_scale */
        std::vector{std::pair{phi_left->children()[0].get(), 1.0},
                    std::pair{phi_left->children()[1].get(), -1.0}}));

    children_.push_back(std::make_shared<ThreePointWaveletFn>(
        /* parent */ this,
        /* index */ 2 * index_ + 1,
        /* single_scale */
        std::vector{std::pair{phi_right->children()[0].get(), 1.0},
                    std::pair{phi_right->children()[1].get(), -1.0}}));
  }
  return true;
}

bool ThreePointWaveletFn::is_full() const {
  if (level_ <= 0)
    return children_.size() == 1;
  else
    return children_.size() == 2;
}

}  // namespace Time
