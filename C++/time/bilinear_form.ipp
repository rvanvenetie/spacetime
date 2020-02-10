#pragma once
#include <iostream>

#include "bilinear_form.hpp"
#include "linear_operator.hpp"

namespace Time {

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
BilinearForm<Operator, I_in, I_out>::BilinearForm(I_in *root_vec_in,
                                                  I_out *root_vec_out)
    : vec_in_(root_vec_in), vec_out_(root_vec_out) {
  assert(vec_in_->is_root());
  assert(vec_out_->is_root());
  // Slice the output vector into levelwise sparse indices.
  for (const auto &node : vec_out_->Bfs()) {
    assert(node->level() >= 0 && node->level() <= lvl_ind_out_.size());
    if (node->level() == lvl_ind_out_.size()) lvl_ind_out_.emplace_back();
    lvl_ind_out_[node->level()].emplace_back(node->node());
  }
  assert(lvl_ind_out_.size());
  assert(lvl_ind_out_[0].size());
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
void BilinearForm<Operator, I_in, I_out>::InitializeInput() {
  // Reset existing input nodes.
  for (auto &sparse_vec : lvl_vec_in_) sparse_vec.clear();

  // Slice the input vector into levelwise sparse vectors.
  for (const auto &node : vec_in_->Bfs()) {
    assert(node->level() >= 0 && node->level() <= lvl_vec_in_.size());
    if (node->level() == lvl_vec_in_.size()) lvl_vec_in_.emplace_back();
    lvl_vec_in_[node->level()].emplace_back(node->node(), node->value());
  }
  assert(lvl_vec_in_.size());
  assert(lvl_vec_in_[0].size());
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
Eigen::MatrixXd BilinearForm<Operator, I_in, I_out>::ToMatrix() {
  auto values_backup = vec_in_->ToVector();
  auto nodes_in = vec_in_->Bfs();
  auto nodes_out = vec_out_->Bfs();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nodes_out.size(), nodes_in.size());
  for (int i = 0; i < nodes_in.size(); ++i) {
    vec_in_->Reset();
    nodes_in[i]->set_value(1);
    Apply();
    for (int j = 0; j < nodes_out.size(); ++j) {
      A(j, i) = nodes_out[j]->value();
    }
  }
  vec_in_->FromVector(values_backup);
  return A;
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::ApplyRecur(
    size_t l, const SparseIndices<ScalingBasisOut> &Pi_out,
    const SparseVector<ScalingBasisIn> &d)
    -> std::pair<SparseVector<ScalingBasisOut>, SparseVector<WaveletBasisOut>> {
  if (l < lvl_vec_in_.size()) lvl_vec_in_.resize(l);
  const SparseVector<WaveletBasisIn> &c = lvl_vec_in_[l];

  if (l < lvl_ind_out_.size()) lvl_ind_out_.resize(l);
  const SparseIndices<WaveletBasisOut> &Lambda_l_out = lvl_ind_out_[l];

  SparseIndices<ScalingBasisIn> Pi_in = d.Indices();
  if ((Pi_out.size() + Lambda_l_out.size()) > 0 &&
      (Pi_in.size() + c.size()) > 0) {
    auto [Pi_B_out, Pi_A_out] = ConstructPiOut(Pi_out);
    auto [Pi_B_in, Pi_A_in] = ConstructPiIn(Pi_in, Pi_B_out);

    auto d_bar = Prolongate<ScalingBasisIn>().MatVec(d.Restrict(Pi_B_in));
    d_bar += WaveletToScaling<WaveletBasisIn>().MatVec(c);

    auto Pi_bar_out = Prolongate<ScalingBasisOut>().Range(Pi_B_out);
    Pi_bar_out |= WaveletToScaling<WaveletBasisOut>().Range(Lambda_l_out);

    auto [e_bar, f_bar] = ApplyRecur(l + 1, Pi_bar_out, d_bar);

    auto e = Operator<ScalingBasisIn, ScalingBasisOut>().MatVec(d, Pi_A_out);
    e += Prolongate<ScalingBasisOut>().RMatVec(e_bar, Pi_B_out);

    auto f = WaveletToScaling<WaveletBasisOut>().RMatVec(e_bar, Lambda_l_out);
    // We know that f and f_bar are disjoint; we can simply append f_bar to f.
    f.insert(f.end(), f_bar.begin(), f_bar.end());
    return std::pair{std::move(e), std::move(f)};
  } else {
    return std::pair{SparseVector<ScalingBasisOut>(),
                     SparseVector<WaveletBasisOut>()};
  }
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::ApplyUppRecur(
    size_t l, const SparseIndices<ScalingBasisOut> &Pi_out,
    const SparseVector<ScalingBasisIn> &d)
    -> std::pair<SparseVector<ScalingBasisOut>, SparseVector<WaveletBasisOut>> {
  if (l < lvl_vec_in_.size()) lvl_vec_in_.resize(l);
  const SparseVector<WaveletBasisIn> &c = lvl_vec_in_[l];

  if (l < lvl_ind_out_.size()) lvl_ind_out_.resize(l);
  const SparseIndices<WaveletBasisOut> &Lambda_l_out = lvl_ind_out_[l];

  SparseIndices<ScalingBasisIn> Pi_in = d.Indices();
  if ((Pi_out.size() + Lambda_l_out.size()) > 0 &&
      (Pi_in.size() + c.size()) > 0) {
    auto d_bar = WaveletToScaling<WaveletBasisIn>().MatVec(c);

    auto [Pi_B_out, Pi_A_out] = ConstructPiOut(Pi_out);
    auto Pi_bar_out = Prolongate<ScalingBasisOut>().Range(Pi_B_out);
    Pi_bar_out |= WaveletToScaling<WaveletBasisOut>().Range(Lambda_l_out);

    auto [e_bar, f_bar] = ApplyUppRecur(l + 1, Pi_bar_out, d_bar);

    auto e = Operator<ScalingBasisIn, ScalingBasisOut>().MatVec(d, Pi_out);
    e += Prolongate<ScalingBasisOut>().RMatVec(e_bar, Pi_B_out);

    auto f = WaveletToScaling<WaveletBasisOut>().RMatVec(e_bar, Lambda_l_out);
    // We know that f and f_bar are disjoint; we can simply append f_bar to f.
    f.insert(f.end(), f_bar.begin(), f_bar.end());
    return std::pair{std::move(e), std::move(f)};
  } else {
    return std::pair{SparseVector<ScalingBasisOut>(),
                     SparseVector<WaveletBasisOut>()};
  }
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::ApplyLowRecur(
    size_t l, const SparseVector<ScalingBasisIn> &d)
    -> SparseVector<WaveletBasisOut> {
  if (l < lvl_vec_in_.size()) lvl_vec_in_.resize(l);
  const SparseVector<WaveletBasisIn> &c = lvl_vec_in_[l];

  if (l < lvl_ind_out_.size()) lvl_ind_out_.resize(l);
  const SparseIndices<WaveletBasisOut> &Lambda_l_out = lvl_ind_out_[l];

  SparseIndices<ScalingBasisIn> Pi_in = d.Indices();
  if (Lambda_l_out.size() > 0 && (Pi_in.size() + c.size()) > 0) {
    auto [Pi_B_in, _] = ConstructPiIn(Pi_in, {});
    auto Pi_B_bar_out = WaveletToScaling<WaveletBasisOut>().Range(Lambda_l_out);

    auto d_bar = Prolongate<ScalingBasisIn>().MatVec(d.Restrict(Pi_B_in));
    auto e_bar =
        Operator<ScalingBasisIn, ScalingBasisOut>().MatVec(d_bar, Pi_B_bar_out);
    d_bar += WaveletToScaling<WaveletBasisIn>().MatVec(c);
    auto f_bar = ApplyLowRecur(l + 1, d_bar);
    auto f = WaveletToScaling<WaveletBasisOut>().RMatVec(e_bar, Lambda_l_out);
    // We know that f and f_bar are disjoint; we can simply append f_bar to f.
    f.insert(f.end(), f_bar.begin(), f_bar.end());
    return f;
  } else {
    return SparseVector<WaveletBasisOut>();
  }
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::ConstructPiOut(
    const SparseIndices<ScalingBasisOut> &Pi_out)
    -> std::pair<SparseIndices<ScalingBasisOut>,
                 SparseIndices<ScalingBasisOut>> {
  SparseIndices<ScalingBasisOut> Pi_A_out, Pi_B_out;
  if (Pi_out.empty()) return {{}, {}};

  int level = Pi_out[0]->level();
  if (level + 1 >= lvl_vec_in_.size() || lvl_vec_in_[level + 1].empty())
    return {{}, Pi_out};

  // Mark the support of wavelets psi, on one level higher.
  auto wavelets = lvl_vec_in_.at(level + 1).Indices();
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

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::ConstructPiIn(
    const SparseIndices<ScalingBasisIn> &Pi_in,
    const SparseIndices<ScalingBasisOut> &Pi_B_out)
    -> std::pair<SparseIndices<ScalingBasisIn>, SparseIndices<ScalingBasisIn>> {
  SparseIndices<ScalingBasisIn> Pi_A_in, Pi_B_in;
  if (Pi_in.empty()) return {{}, {}};
  int level = Pi_in[0]->level();

  // Mark the support of wavelets psi, on one level higher.
  SparseIndices<WaveletBasisOut> wavelets;
  if (level + 1 < lvl_ind_out_.size()) wavelets = lvl_ind_out_.at(level + 1);

  for (auto psi : wavelets)
    for (auto elem : psi->support()) elem->parent()->set_marked(true);

  // Mark the support of scaling functions in Pi_B_out.
  for (auto phi : Pi_B_out)
    for (auto elem : phi->support()) elem->set_marked(true);

  for (auto phi : Pi_in)
    if (std::any_of(phi->support().begin(), phi->support().end(),
                    [](auto elem) { return elem->marked(); }))
      Pi_B_in.emplace_back(phi);
    else
      Pi_A_in.emplace_back(phi);

  // Unmark.
  for (auto psi : wavelets)
    for (auto elem : psi->support()) elem->parent()->set_marked(false);
  for (auto phi : Pi_B_out)
    for (auto elem : phi->support()) elem->set_marked(false);

  return {std::move(Pi_B_in), std::move(Pi_A_in)};
}

}  // namespace Time
