#pragma once
#include <memory>
#include <vector>

#include "basis.hpp"

namespace Time {

template <>
struct FunctionTrait<OrthonormalWaveletFn> {
  using Scaling = DiscLinearScalingFn;
};

class DiscLinearScalingFn : public ScalingFn<DiscLinearScalingFn> {
 public:
  constexpr static size_t order = 1;
  constexpr static bool continuous = false;
  constexpr static const char *name = "DLS";

  explicit DiscLinearScalingFn(const std::vector<DiscLinearScalingFn *> parents,
                               int index,
                               const std::vector<Element1D *> support)
      : ScalingFn<DiscLinearScalingFn>(parents, index, support) {
    if (pw_constant())
      support[0]->phi_disc_lin_[0] = this;
    else
      support[0]->phi_disc_lin_[1] = this;
  }

  inline bool pw_constant() const { return index() % 2 == 0; }

  double Eval(double t, bool deriv = false) const;
  double EvalMother(double t, bool deriv) const;
  bool Refine();

 protected:
  DiscLinearScalingFn *nbr_ = nullptr;
  inline bool is_full() const {
    if (is_metaroot())
      return children_.size() == 2;
    else
      return children_.size() == 4;
  }

  // Protected constructor for creating a metaroot.
  DiscLinearScalingFn(
      datastructures::TreeContainer<DiscLinearScalingFn> *container,
      Element1D *mother_element);

  friend datastructures::Tree<DiscLinearScalingFn>;
  friend OrthonormalWaveletFn;
};

class OrthonormalWaveletFn : public WaveletFn<OrthonormalWaveletFn> {
 public:
  constexpr static const char *name = "Ortho";

  explicit OrthonormalWaveletFn(
      const std::vector<OrthonormalWaveletFn *> parents, int index,
      SparseVector<DiscLinearScalingFn> &&single_scale)
      : WaveletFn(parents, index, std::move(single_scale)) {
    for (auto &elem : support()) {
      assert(elem->psi_ortho_[index % 2] == nullptr);
      elem->psi_ortho_[index % 2] = this;
    }
  }

  bool Refine();
  bool is_full() const;

 protected:
  // Protected constructor for creating a metaroot.
  OrthonormalWaveletFn(
      datastructures::TreeContainer<OrthonormalWaveletFn> *container,
      const SmallVector<
          DiscLinearScalingFn *,
          datastructures::NodeTrait<DiscLinearScalingFn>::N_children>
          &mother_scalings);

  friend datastructures::Tree<OrthonormalWaveletFn>;
};

}  // namespace Time
