#include "haar_basis.hpp"

#include <iostream>
#include <memory>
namespace Time {

// Metaroot constructor.
DiscConstantScalingFn::DiscConstantScalingFn(
    Deque<DiscConstantScalingFn> *container, Element1D *mother_element)
    : ScalingFn<DiscConstantScalingFn>(container) {
  make_child(/* parent */ this, /* index */ 0, /* support */ mother_element);
}

double DiscConstantScalingFn::EvalMother(double t, bool deriv) const {
  if (deriv || t < 0 || t >= 1)
    return 0;
  else
    return 1;
}

bool DiscConstantScalingFn::Refine() {
  if (is_full()) return false;
  assert(children_.empty());

  support_[0]->Refine();
  auto [l, n] = labda();

  // clang-format off
  make_child(
      /* parent */ this,
      /* index */ 2 * n,
      /* support */ support_[0]->children()[0]);
  make_child(
      /* parent */ this,
      /* index */ 2 * n + 1,
      /* support */ support_[0]->children()[1]);
  // clang-format on

  return true;
}

HaarWaveletFn::HaarWaveletFn(Deque<HaarWaveletFn> *container,
                             DiscConstantScalingFn *mother_scaling)
    : WaveletFn<HaarWaveletFn>(container) {
  make_child(/* parent */ this, /* index */ 0,
             /* support */ std::vector{std::pair{mother_scaling, 1.0}});
}

bool HaarWaveletFn::Refine() {
  if (is_full()) return true;
  assert(children_.empty());

  if (level_ == 0) {
    auto mother_scaling = single_scale_[0].first;
    mother_scaling->Refine();
    auto mother_scaling_children = mother_scaling->children();

    make_child(
        /* parent */ this,
        /* index */ 0,
        /* single_scale */
        std::vector{std::pair{mother_scaling_children[0], 1.0},
                    std::pair{mother_scaling_children[1], -1.0}});
  } else {
    assert(level_ > 0);
    auto phi_left = single_scale_[0].first;
    auto phi_right = single_scale_[1].first;
    phi_left->Refine();
    phi_right->Refine();

    make_child(
        /* parent */ this,
        /* index */ 2 * index_,
        /* single_scale */
        std::vector{std::pair{phi_left->children()[0], 1.0},
                    std::pair{phi_left->children()[1], -1.0}});

    make_child(
        /* parent */ this,
        /* index */ 2 * index_ + 1,
        /* single_scale */
        std::vector{std::pair{phi_right->children()[0], 1.0},
                    std::pair{phi_right->children()[1], -1.0}});
  }
  return true;
}

bool HaarWaveletFn::is_full() const {
  if (level_ <= 1)
    return children_.size() == 1;
  else
    return children_.size() == 2;
}

}  // namespace Time
