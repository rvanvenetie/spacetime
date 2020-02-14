#pragma once

#include "bilinear_form.hpp"
#include "linear_form.hpp"

namespace Time {

template <typename WaveletBasis>
template <typename I>
void LinearForm<WaveletBasis>::Apply(I *root) {
  assert(root->is_root());
  lvl_ind_out_.clear();
  auto nodes_vec_out = root->Bfs();
  for (const auto &node : nodes_vec_out) {
    assert(node->level() >= 0 && node->level() <= lvl_ind_out_.size());
    if (node->level() == lvl_ind_out_.size()) lvl_ind_out_.emplace_back();
    lvl_ind_out_[node->level()].emplace_back(node->node());
  }

  auto [_, f] = ApplyRecur(0, {});

  f.StoreInTree();
  for (const auto &nv : nodes_vec_out)
    nv->set_value(*nv->node()->template data<double>());
  f.RemoveFromTree();
}

template <typename WaveletBasis>
auto LinearForm<WaveletBasis>::ApplyRecur(
    size_t l, const SparseIndices<ScalingBasis> &Pi_out)
    -> std::pair<SparseVector<ScalingBasis>, SparseVector<WaveletBasis>> {
  const SparseIndices<WaveletBasis> &Lambda_l_out =
      l < lvl_ind_out_.size() ? lvl_ind_out_[l] : empty_ind_out_;

  if ((Pi_out.size() + Lambda_l_out.size()) > 0) {
    auto [Pi_B_out, Pi_A_out] = ConstructPiOut(Pi_out);
    auto Pi_bar_out = Prolongate<ScalingBasis>().Range(Pi_B_out);
    Pi_bar_out |= WaveletToScaling<WaveletBasis>().Range(Lambda_l_out);

    auto [e_bar, f_bar] = ApplyRecur(l + 1, Pi_bar_out);

    auto e = functional_->Eval(Pi_A_out);
    e += Prolongate<ScalingBasis>().RMatVec(e_bar, Pi_B_out);

    auto f = WaveletToScaling<WaveletBasis>().RMatVec(e_bar, Lambda_l_out);
    return std::pair{std::move(e), Union(std::move(f_bar), std::move(f))};
  }
  return std::pair{SparseVector<ScalingBasis>(), SparseVector<WaveletBasis>()};
}

template <typename WaveletBasis>
auto LinearForm<WaveletBasis>::ConstructPiOut(
    const SparseIndices<ScalingBasis> &Pi_out)
    -> std::pair<SparseIndices<ScalingBasis>, SparseIndices<ScalingBasis>> {
  SparseIndices<ScalingBasis> Pi_A_out, Pi_B_out;
  if (Pi_out.empty()) return {{}, {}};

  int level = Pi_out[0]->level();
  if (level + 1 >= lvl_ind_out_.size()) return {{}, Pi_out};

  // Mark the support of wavelets psi, on one level higher.
  auto &wavelets = lvl_ind_out_.at(level + 1);
  for (auto psi : wavelets)
    for (auto elem : psi->support()) elem->parent()->set_marked(true);

  for (auto phi : Pi_out)
    if (std::any_of(phi->support().begin(), phi->support().end(),
                    [](auto elem) { return elem->marked(); }))
      Pi_B_out.emplace_back(phi);
    else
      Pi_A_out.emplace_back(phi);

  // Unmark.
  for (auto psi : wavelets)
    for (auto elem : psi->support()) elem->parent()->set_marked(false);

  return {std::move(Pi_B_out), std::move(Pi_A_out)};
}
};  // namespace Time
