#include "three_point_basis.hpp"

#include <iostream>
#include <memory>
namespace Time {

// Initialize static variables.
datastructures::Tree<ContLinearScalingFn> cont_lin_tree;
datastructures::Tree<ThreePointWaveletFn> three_point_tree;

// Metaroot constructor.
ContLinearScalingFn::ContLinearScalingFn() : ScalingFn<ContLinearScalingFn>() {
  auto scaling_left =
      std::make_shared<ContLinearScalingFn>({this}, 0, {mother_element});
  auto scaling_right =
      std::make_shared<ContLinearScalingFn>({this}, 1, {mother_element});

  scaling_left->nbr_right_ = scaling_right;
  scaling_right->nbr_left_ = scaling_left;
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
  if (child_mid_) return child_mid_;
  support_[0]->Refine();
  auto [l, n] = labda();
  for (auto elem : support_) elem->Refine();

  std::vector<Element1D *> child_support;
  if (n > 0) child_support.push_back(support_[0]->children()[1]);

  if
    n > 0 : child_support.append(self.support[0].children[1]) if n < 2 * *l
        : child_support.append(self.support[-1].children[0])

#Create element.
              child = ContLinearScaling((l + 1, 2 * n), child_support, [self])
                          self.child_mid = child self
                                               ._update_children()

#Update nbrs.
                                                   if self.child_left
        : self.child_left.nbr_right = child child.nbr_left =
        self.child_left

        if self.child_right : self.child_right.nbr_left =
            child child.nbr_right = self.child_right

                                    return child assert(children_.empty());

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
  for (auto scaling : cont_lin_tree.meta_root->children()) {
    children_.push_back(std::make_shared<ThreePointWaveletFn>(
        {this}, 0, std::vector{std::pair{scaling.get(), 1.0}}));
  }
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
