#include "orthonormal_basis.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <memory>

namespace Time {

DiscLinearScalingFn::DiscLinearScalingFn(Element1D *mother_element)
    : ScalingFn<DiscLinearScalingFn>() {
  auto scaling_left = make_child(
      /* parents */ std::vector{this},
      /* index */ 0,
      /* support */ std::vector{mother_element});
  auto scaling_right = make_child(
      /* parents */ std::vector{this},
      /* index */ 1,
      /* support */ std::vector{mother_element});
  scaling_left->nbr_ = scaling_right;
  scaling_right->nbr_ = scaling_left;
}

double DiscLinearScalingFn::EvalMother(double t, bool deriv) const {
  double mask = (0 <= t) && (t < 1);
  if (!deriv) {
    if (pw_constant())
      return mask;
    else
      return sqrt(3) * (2 * t - 1) * mask;
  } else {
    if (pw_constant())
      return 0.0;
    else
      return sqrt(3) * 2.0 * mask;
  }
}

double DiscLinearScalingFn::Eval(double t, bool deriv) const {
  auto [l, n] = labda();
  double chain_rule_constant = deriv ? (1 << l) : 1;
  return chain_rule_constant * EvalMother((1 << l) * t - (n / 2), deriv);
}

bool DiscLinearScalingFn::Refine() {
  if (is_full()) return false;
  assert(children_.empty());
  support_[0]->Refine();
  auto [l, n] = labda();
  auto P = std::vector{this, nbr_};
  auto child_elts = support_[0]->children();
  make_child(
      /* parents */ P, /* index */ 2 * n + 0,
      /* support */ std::vector{child_elts[0]});
  make_child(
      /* parents */ P, /* index */ 2 * n + 1,
      /* support */ std::vector{child_elts[0]});
  make_child(
      /* parents */ P, /* index */ 2 * n + 2,
      /* support */ std::vector{child_elts[1]});
  make_child(
      /* parents */ P, /* index */ 2 * n + 3,
      /* support */ std::vector{child_elts[1]});

  nbr_->children_ = children_;
  children_[0]->nbr_ = children_[1];
  children_[1]->nbr_ = children_[0];
  children_[2]->nbr_ = children_[3];
  children_[3]->nbr_ = children_[2];
  return true;
}

OrthonormalWaveletFn::OrthonormalWaveletFn(
    const SmallVector<
        DiscLinearScalingFn *,
        datastructures::NodeTrait<DiscLinearScalingFn>::N_children>
        &mother_scalings)
    : WaveletFn<OrthonormalWaveletFn>() {
  assert(mother_scalings.size() == 2);
  for (size_t i = 0; i < 2; ++i) {
    make_child(
        /* parents */ std::vector{this},
        /* index */ i,
        /* single_scale */ std::vector{std::pair{mother_scalings[i], 1.0}});
  }
}

bool OrthonormalWaveletFn::Refine() {
  if (is_full()) return true;
  assert(children_.empty());
  if (level_ == 0) {
    auto parents = this->parents()[0]->children();
    assert(parents.size() == 2);

    single_scale_[0].first->Refine();
    auto l1_scalings = single_scale_[0].first->children();

    make_child(
        /* parents */ std::vector{parents[0], parents[1]},
        /* index */ 0,
        /* single_scale */
        std::vector{std::pair{l1_scalings[0], -1.0 / 2},
                    std::pair{l1_scalings[1], -sqrt(3) / 2},
                    std::pair{l1_scalings[2], 1.0 / 2},
                    std::pair{l1_scalings[3], -sqrt(3) / 2}});
    make_child(
        /* parents */ std::vector{parents[0], parents[1]},
        /* index */ 1,
        /* single_scale */
        std::vector{std::pair{l1_scalings[1], -1.0},
                    std::pair{l1_scalings[3], 1.0}});
    if (index_ == 0)
      parents[1]->children_ = children_;
    else
      parents[0]->children_ = children_;
  } else {
    auto [l, n] = labda();
    constexpr std::array<int, 4> nbr_indices{1, 0, 3, 2};
    auto nbr = parents()[0]->children()[nbr_indices[n % 4]];
    assert(support_ == nbr->support_);
    assert(labda() != nbr->labda());

    if (n % 2) return nbr->Refine();
    auto s = std::pow(2.0, l / 2.0);
    for (int i = 0; i < 2; i++) {
      auto phi = single_scale_[2 * i].first;
      phi->Refine();
      auto parents = std::vector{this, nbr};
      make_child(
          /* parents */ parents, /* index */ 2 * (n + i),
          /* single_scale */
          std::vector{std::pair{phi->children_[0], -s / 2},
                      std::pair{phi->children_[1], -s * sqrt(3) / 2},
                      std::pair{phi->children_[2], s / 2},
                      std::pair{phi->children_[3], -s * sqrt(3) / 2}});
      make_child(
          /* parents */ parents,
          /* index */ 2 * (n + i) + 1,
          /* single_scale */
          std::vector{std::pair{phi->children_[1], -s},
                      std::pair{phi->children_[3], s}});
    }
    nbr->children_ = children_;
  }
  return true;
}

bool OrthonormalWaveletFn::is_full() const {
  if (level_ <= 0)
    return children_.size() == 2;
  else
    return children_.size() == 4;
}

}  // namespace Time
