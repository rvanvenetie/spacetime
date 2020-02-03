#pragma once
#include <memory>
#include <vector>

#include "basis.hpp"

namespace Time {
class DiscLinearScalingFn;
class OrthonormalWaveletFn;

template <>
struct FunctionTrait<DiscLinearScalingFn> {
  using Wavelet = OrthonormalWaveletFn;
};

template <>
struct FunctionTrait<OrthonormalWaveletFn> {
  using Scaling = DiscLinearScalingFn;
};

class DiscLinearScalingFn : public ScalingFn<DiscLinearScalingFn> {
 public:
  constexpr static size_t order = 1;
  constexpr static bool continuous = false;
  constexpr static size_t N_children = 4;
  constexpr static size_t N_parents = 2;
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
  DiscLinearScalingFn();

  friend datastructures::Tree<DiscLinearScalingFn>;
  friend OrthonormalWaveletFn;
};

class OrthonormalWaveletFn : public WaveletFn<OrthonormalWaveletFn> {
 public:
  constexpr static size_t N_children = 4;
  constexpr static size_t N_parents = 2;
  constexpr static const char *name = "Ortho";

  explicit OrthonormalWaveletFn(
      const std::vector<OrthonormalWaveletFn *> parents, int index,
      const SparseVector<DiscLinearScalingFn> &single_scale)
      : WaveletFn(parents, index, single_scale) {
    for (auto &elem : support()) {
      assert(elem->psi_ortho_[index % 2] == nullptr);
      elem->psi_ortho_[index % 2] = this;
    }
  }

  bool Refine();
  bool is_full() const;

 protected:
  // Protected constructor for creating a metaroot.
  OrthonormalWaveletFn();

  friend datastructures::Tree<OrthonormalWaveletFn>;
};

// Define static variables.
extern datastructures::Tree<DiscLinearScalingFn> disc_lin_tree;
extern datastructures::Tree<OrthonormalWaveletFn> ortho_tree;

}  // namespace Time
