#pragma once
#include <iostream>

#include "bilinear_form.hpp"
#include "linear_operator.hpp"

namespace Time {

// Optimized helper function for unioning two disjoint basisvectors.
template <typename Basis>
inline SparseVector<Basis> Union(SparseVector<Basis> &&a,
                                 SparseVector<Basis> &&b) {
  // Ensure that a.size() <= b.size().
  if (b.size() > a.size()) return Union(std::move(b), std::move(a));
  if (b.empty()) return std::move(a);
  a.reserve(a.size() + b.size());
  a.insert(a.end(), std::make_move_iterator(b.begin()),
           std::make_move_iterator(b.end()));
  return std::move(a);
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
BilinearForm<Operator, I_in, I_out>::BilinearForm(I_in *root_vec_in,
                                                  I_out *root_vec_out)
    : vec_in_(root_vec_in),
      vec_out_(root_vec_out),
      nodes_vec_in_(std::make_shared<std::vector<std::vector<I_in *>>>(
          vec_in_->NodesPerLevel())),
      nodes_vec_out_(std::make_shared<std::vector<std::vector<I_out *>>>(
          vec_out_->NodesPerLevel())) {}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::LevelVectorInput(size_t l) const
    -> SparseVector<WaveletBasisIn> {
  SparseVector<WaveletBasisIn> result;
  if (l >= nodes_vec_in_->size()) return result;
  const auto &nodes_lvl = (*nodes_vec_in_)[l];
  result.reserve(nodes_lvl.size());
  for (const auto &nv : nodes_lvl) result.emplace_back(nv->node(), nv->value());
  return result;
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::LevelIndicesOutput(size_t l) const
    -> SparseIndices<WaveletBasisOut> {
  SparseIndices<WaveletBasisOut> result;
  if (l >= nodes_vec_out_->size()) return result;
  const auto &nodes_lvl = (*nodes_vec_out_).at(l);
  result.reserve(nodes_lvl.size());
  for (const auto &nv : nodes_lvl) result.emplace_back(nv->node());
  return result;
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
void BilinearForm<Operator, I_in, I_out>::FinalizeOutput(
    const SparseVector<WaveletBasisOut> &f) {
  // Store the sparse vector in the underlying tree.
  f.StoreInTree();

  // Copy the values in the underlying tree to the output vector.
  for (const auto &nodes_lvl : *nodes_vec_out_)
    for (const auto nv : nodes_lvl) {
      auto node = nv->node();
      if (node->has_data())
        nv->set_value(*node->template data<double>());
      else
        nv->set_value(0);
    }

  // Remove the data in the underlying tree.
  f.RemoveFromTree();
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
    size_t l, SparseIndices<ScalingBasisOut> &&Pi_out,
    const SparseVector<ScalingBasisIn> &d)
    -> std::pair<SparseVector<ScalingBasisOut>, SparseVector<WaveletBasisOut>> {
  SparseVector<WaveletBasisIn> c = LevelVectorInput(l);
  SparseIndices<WaveletBasisOut> Lambda_l_out = LevelIndicesOutput(l);

  if ((Pi_out.size() + Lambda_l_out.size()) > 0 && (d.size() + c.size()) > 0) {
    auto [Pi_B_out, Pi_A_out] = ConstructPiOut(std::move(Pi_out));
    auto Pi_B_in = ConstructPiBIn(d.Indices(), Pi_B_out);

    auto d_bar = WaveletToScaling<WaveletBasisIn>().MatVec(c);
    d_bar += Prolongate<ScalingBasisIn>().MatVec(d.Restrict(Pi_B_in));

    auto Pi_bar_out = Prolongate<ScalingBasisOut>().Range(Pi_B_out);
    Pi_bar_out |= WaveletToScaling<WaveletBasisOut>().Range(Lambda_l_out);

    auto [e_bar, f_bar] = ApplyRecur(l + 1, std::move(Pi_bar_out), d_bar);

    auto e = Operator<ScalingBasisIn, ScalingBasisOut>().MatVec(d, Pi_A_out);
    e += Prolongate<ScalingBasisOut>().RMatVec(e_bar, Pi_B_out);

    auto f = WaveletToScaling<WaveletBasisOut>().RMatVec(e_bar, Lambda_l_out);
    return std::pair{std::move(e), Union(std::move(f_bar), std::move(f))};
  } else {
    return std::pair{SparseVector<ScalingBasisOut>(),
                     SparseVector<WaveletBasisOut>()};
  }
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::ApplyUppRecur(
    size_t l, SparseIndices<ScalingBasisOut> &&Pi_out,
    const SparseVector<ScalingBasisIn> &d)
    -> std::pair<SparseVector<ScalingBasisOut>, SparseVector<WaveletBasisOut>> {
  SparseVector<WaveletBasisIn> c = LevelVectorInput(l);
  SparseIndices<WaveletBasisOut> Lambda_l_out = LevelIndicesOutput(l);

  if ((Pi_out.size() + Lambda_l_out.size()) > 0 && (d.size() + c.size()) > 0) {
    auto d_bar = WaveletToScaling<WaveletBasisIn>().MatVec(c);
    auto e = Operator<ScalingBasisIn, ScalingBasisOut>().MatVec(d, Pi_out);

    auto [Pi_B_out, _] =
        ConstructPiOut(std::move(Pi_out), /* construct_Pi_A_out */ false);
    auto Pi_bar_out = Prolongate<ScalingBasisOut>().Range(Pi_B_out);
    Pi_bar_out |= WaveletToScaling<WaveletBasisOut>().Range(Lambda_l_out);

    auto [e_bar, f_bar] = ApplyUppRecur(l + 1, std::move(Pi_bar_out), d_bar);

    e += Prolongate<ScalingBasisOut>().RMatVec(e_bar, Pi_B_out);

    auto f = WaveletToScaling<WaveletBasisOut>().RMatVec(e_bar, Lambda_l_out);
    return std::pair{std::move(e), Union(std::move(f_bar), std::move(f))};
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
  SparseVector<WaveletBasisIn> c = LevelVectorInput(l);
  SparseIndices<WaveletBasisOut> Lambda_l_out = LevelIndicesOutput(l);

  if (Lambda_l_out.size() > 0 && (d.size() + c.size()) > 0) {
    auto Pi_B_in = ConstructPiBIn(d.Indices(), {});
    auto Pi_B_bar_out = WaveletToScaling<WaveletBasisOut>().Range(Lambda_l_out);

    auto d_bar = Prolongate<ScalingBasisIn>().MatVec(d.Restrict(Pi_B_in));
    auto e_bar =
        Operator<ScalingBasisIn, ScalingBasisOut>().MatVec(d_bar, Pi_B_bar_out);
    d_bar += WaveletToScaling<WaveletBasisIn>().MatVec(c);
    auto f_bar = ApplyLowRecur(l + 1, d_bar);
    auto f = WaveletToScaling<WaveletBasisOut>().RMatVec(e_bar, Lambda_l_out);
    return Union(std::move(f_bar), std::move(f));
  } else {
    return SparseVector<WaveletBasisOut>();
  }
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::ConstructPiOut(
    SparseIndices<ScalingBasisOut> &&Pi_out, bool construct_Pi_A_out)
    -> std::pair<SparseIndices<ScalingBasisOut>,
                 SparseIndices<ScalingBasisOut>> {
  if (Pi_out.empty()) return {{}, {}};

  int level = Pi_out[0]->level();
  if (level + 1 >= nodes_vec_in_->size() || (*nodes_vec_in_)[level + 1].empty())
    return {{}, std::move(Pi_out)};

  // Mark the support of wavelets psi, on one level higher.
  const auto &wavelets = (*nodes_vec_in_)[level + 1];
  for (const auto &psi : wavelets)
    for (auto elem : psi->node()->support()) elem->parent()->set_marked(true);

  SparseIndices<ScalingBasisOut> Pi_B_out, Pi_A_out;
  Pi_B_out.reserve(Pi_out.size());
  if (construct_Pi_A_out) Pi_A_out.reserve(Pi_out.size());

  // Loop over Pi_out and assign the fn to the correct output set.
  for (auto phi : Pi_out)
    if (std::any_of(phi->support().begin(), phi->support().end(),
                    [](auto elem) { return elem->marked(); }))
      Pi_B_out.emplace_back(phi);
    else if (construct_Pi_A_out)
      Pi_A_out.emplace_back(phi);

  // Unmark.
  for (const auto &psi : wavelets)
    for (auto elem : psi->node()->support()) elem->parent()->set_marked(false);

  return {std::move(Pi_B_out), std::move(Pi_A_out)};
}

template <template <typename, typename> class Operator, typename I_in,
          typename I_out>
auto BilinearForm<Operator, I_in, I_out>::ConstructPiBIn(
    SparseIndices<ScalingBasisIn> &&Pi_in,
    const SparseIndices<ScalingBasisOut> &Pi_B_out)
    -> SparseIndices<ScalingBasisIn> {
  if (Pi_in.empty()) return {};
  int level = Pi_in[0]->level();

  // Mark the support of wavelets psi, on one level higher.
  if (level + 1 < nodes_vec_out_->size())
    for (auto psi : (*nodes_vec_out_)[level + 1])
      for (auto elem : psi->node()->support()) elem->parent()->set_marked(true);

  // Mark the support of scaling functions in Pi_B_out.
  for (auto phi : Pi_B_out)
    for (auto elem : phi->support()) elem->set_marked(true);

  // Set Pi_B_in to whole Pi_in.
  SparseIndices<ScalingBasisIn> Pi_B_in(std::move(Pi_in));

  // Now remove all fn from Pi_B_in that have no marked element.
  Pi_B_in.erase(std::remove_if(Pi_B_in.begin(), Pi_B_in.end(),
                               [](auto phi) {
                                 for (auto elem : phi->support())
                                   if (elem->marked()) return false;
                                 return true;
                               }),
                Pi_B_in.end());

  // Unmark.
  if (level + 1 < nodes_vec_out_->size())
    for (auto psi : (*nodes_vec_out_)[level + 1])
      for (auto elem : psi->node()->support())
        elem->parent()->set_marked(false);
  for (auto phi : Pi_B_out)
    for (auto elem : phi->support()) elem->set_marked(false);

  return Pi_B_in;
}

}  // namespace Time
