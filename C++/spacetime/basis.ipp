#pragma once
#include <vector>

#include "basis.hpp"

namespace spacetime {
template <typename DblTreeIn>
datastructures::DoubleTreeView<Time::OrthonormalWaveletFn,
                               space::HierarchicalBasisFn>
GenerateYDelta(const DblTreeIn &X_delta) {
  using SpaceTreeVector = std::vector<typename DblTreeIn::FrozenDN1Type *>;
  auto Y_delta = datastructures::DoubleTreeView<Time::OrthonormalWaveletFn,
                                                space::HierarchicalBasisFn>(
      Time::ortho_tree.meta_root.get(), std::get<1>(X_delta.root()->nodes()));

  Y_delta.Project_1()->Union(X_delta.Project_1());

  for (auto &x_labda_0 : X_delta.Project_0()->Bfs())
    for (auto &elem : x_labda_0->node()->support())
      for (auto &mu : elem->RefinePsiOrthonormal()) {
        if (!mu->has_data()) mu->set_data(new SpaceTreeVector());
        mu->template data<SpaceTreeVector>()->push_back(
            x_labda_0->FrozenOtherAxis());
      }

  Y_delta.Project_0()->DeepRefine([](auto node) { return node->has_data(); });

  for (auto &y_labda_0 : Y_delta.Project_0()->Bfs()) {
    assert(y_labda_0->node()->has_data());
    auto data = y_labda_0->node()->template data<SpaceTreeVector>();
    for (const auto &space_tree : *data)
      y_labda_0->FrozenOtherAxis()->Union(space_tree);
    y_labda_0->node()->reset_data();
    delete data;
  }

  return Y_delta;
}

template <typename DblTreeIn>
datastructures::DoubleTreeView<Time::ThreePointWaveletFn,
                               space::HierarchicalBasisFn>
GenerateXDeltaUnderscore(const DblTreeIn &X_delta, size_t num_repeats) {
  assert(num_repeats > 0);
  if (num_repeats > 1)
    return GenerateXDeltaUnderscore(GenerateXDeltaUnderscore(X_delta),
                                    num_repeats - 1);
  auto X_delta_underscore =
      X_delta.template DeepCopy<datastructures::DoubleTreeView<
          Time::ThreePointWaveletFn, space::HierarchicalBasisFn>>();
  std::vector<datastructures::DoubleNodeView<Time::ThreePointWaveletFn,
                                             space::HierarchicalBasisFn> *>
      time_leaves, space_leaves;
  for (auto &dblnode : X_delta_underscore.container()) {
    auto [time_node, space_node] = dblnode.nodes();
    if (!time_node->is_full()) time_node->Refine();
    if (!space_node->is_full())  // This is DIFFERENT from python.
      space_node->Refine();

    if (!dblnode.template is_full<0>()) time_leaves.push_back(&dblnode);
    if (!dblnode.template is_full<1>()) space_leaves.push_back(&dblnode);
  }

  for (auto dblnode : time_leaves)
    dblnode->Refine<0>(datastructures::func_true, /*make_conforming*/ true);

  for (auto dblnode : space_leaves) {
    dblnode->Refine<1>(datastructures::func_true, /*make_conforming*/ true);
    for (auto child : dblnode->children(1)) {
      child->node_1()->Refine();
      child->Refine<1>(datastructures::func_true, /*make_conforming*/ true);
    }
  }

  return X_delta_underscore;
}

template <class DblTreeIn, class DblTreeOut>
auto GenerateSigma(const DblTreeIn &Lambda_in, const DblTreeOut &Lambda_out) {
  using OutNodeVector = std::vector<typename DblTreeOut::T0 *>;

  for (const auto &psi_out : Lambda_out.Project_0()->Bfs())
    for (auto elem : psi_out->node()->support()) {
      if (!elem->has_data()) elem->set_data(new OutNodeVector());
      elem->template data<OutNodeVector>()->push_back(psi_out->node());
    }

  auto Sigma = std::make_shared<datastructures::DoubleTreeVector<
      typename DblTreeIn::T0, typename DblTreeOut::T1>>(
      std::get<0>(Lambda_in.root()->nodes()),
      std::get<1>(Lambda_out.root()->nodes()));
  Sigma->Project_0()->Union(Lambda_in.Project_0());
  Sigma->Project_1()->Union(Lambda_out.Project_1());

  for (const auto &psi_in_labda_0 : Sigma->Project_0()->Bfs()) {
    // NOTE: The code below is sigma as described in followup3.pdf. We chose
    // to enlarge sigma in order to create a `cheap` transpose of a spacetime
    // bilinear form. To do this, we must add the `diagonal` to Sigma.

    // std::vector<Time::Element1D *> children;
    // children.reserve(psi_in_labda_0->node()->support().size() *
    //                 DblTreeIn::T0::N_children);
    // for (auto elem : psi_in_labda_0->node()->support())
    //  for (auto child : elem->children()) children.push_back(child);

    // std::sort(children.begin(), children.end());
    // auto last = std::unique(children.begin(), children.end());
    // children.erase(last, children.end());

    for (auto child : psi_in_labda_0->node()->support()) {
      if (!child->has_data()) continue;
      for (auto mu : *child->template data<OutNodeVector>())
        psi_in_labda_0->FrozenOtherAxis()->Union(Lambda_out.Fiber_1(mu));
    }
  }

  for (const auto &psi_out : Lambda_out.Project_0()->Bfs())
    for (auto elem : psi_out->node()->support()) {
      if (!elem->has_data()) continue;
      auto data = elem->template data<OutNodeVector>();
      elem->reset_data();
      delete data;
    }

  return Sigma;
}

template <class DblTreeIn, class DblTreeOut>
auto GenerateTheta(const DblTreeIn &Lambda_in, const DblTreeOut &Lambda_out) {
  auto Theta = std::make_shared<datastructures::DoubleTreeVector<
      typename DblTreeOut::T0, typename DblTreeIn::T1>>(
      std::get<0>(Lambda_out.root()->nodes()),
      std::get<1>(Lambda_in.root()->nodes()));
  Theta->Project_0()->Union(Lambda_out.Project_0());
  Theta->Project_1()->Union(Lambda_in.Project_1());

  for (const auto &psi_in_labda_1 : Theta->Project_1()->Bfs()) {
    auto fiber_labda_0 = Lambda_in.Fiber_0(psi_in_labda_1->node());
    auto fiber_labda_0_nodes = fiber_labda_0->Bfs();
    for (const auto &psi_in_labda_0 : fiber_labda_0_nodes)
      for (auto elem : psi_in_labda_0->node()->support())
        elem->set_marked(true);

    psi_in_labda_1->FrozenOtherAxis()->Union(
        Lambda_out.Project_0(), [](const auto &psi_out_labda_0) {
          for (auto elem : psi_out_labda_0->node()->support())
            if (elem->marked()) return true;
          return false;
        });

    for (const auto &psi_in_labda_0 : fiber_labda_0_nodes)
      for (auto elem : psi_in_labda_0->node()->support())
        elem->set_marked(false);
  }
  return Theta;
}
};  // namespace spacetime
