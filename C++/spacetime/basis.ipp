#pragma once
#include "spacetime/basis.hpp"

namespace spacetime {
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
