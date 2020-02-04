#pragma once

namespace spacetime {
template <class DblTreeIn, class DblTreeOut>
auto GenerateSigma(DblTreeIn &Lambda_in, DblTreeOut &Lambda_out) {
  using OutNodeVector = std::vector<typename DblTreeOut::T0p>;

  for (auto &psi_out : Lambda_out.Project_0()->Bfs())
    for (auto &elem : psi_out->node()->support()) {
      if (!elem->has_data()) elem->set_data(new OutNodeVector());
      elem->template data<OutNodeVector>()->push_back(psi_out->node());
    }

  auto Sigma = datastructures::DoubleTreeVector<typename DblTreeIn::T0,
                                                typename DblTreeOut::T1>(
      std::get<0>(Lambda_in.root->nodes()),
      std::get<1>(Lambda_out.root->nodes()));
  Sigma.Project_0()->Union(Lambda_in.Project_0());
  Sigma.Project_1()->Union(Lambda_out.Project_1());

  for (auto &psi_in_labda_0 : Sigma.Project_0()->Bfs()) {
    std::vector<Time::Element1D *> children;
    for (auto &elem : psi_in_labda_0->node()->support())
      for (auto &child : elem->children()) children.push_back(child);

    for (auto &child : children) {
      if (!child->has_data()) continue;
      for (auto &mu : *child->template data<OutNodeVector>())
        psi_in_labda_0->FrozenOtherAxis()->Union(Lambda_out.Fiber_1(mu));
    }
  }

  for (auto &psi_out : Lambda_out.Project_0()->Bfs())
    for (auto &elem : psi_out->node()->support()) {
      if (!elem->has_data()) continue;
      auto data = elem->template data<OutNodeVector>();
      elem->reset_data();
      delete data;
    }

  return Sigma;
}

template <class DblTreeIn, class DblTreeOut>
auto GenerateTheta(DblTreeIn &Lambda_in, DblTreeOut &Lambda_out) {
  using OutNodeVector = std::vector<typename DblTreeOut::T0p>;

  for (auto &psi_out : Lambda_out.Project_0()->Bfs())
    for (auto &elem : psi_out->node()->support()) {
      if (!elem->has_data()) elem->set_data(new OutNodeVector());
      elem->template data<OutNodeVector>()->push_back(psi_out->node());
    }

  auto Theta = datastructures::DoubleTreeVector<typename DblTreeOut::T0,
                                                typename DblTreeIn::T1>(
      std::get<0>(Lambda_out.root->nodes()),
      std::get<1>(Lambda_in.root->nodes()));
  Theta.Project_0()->Union(Lambda_out.Project_0());
  Theta.Project_1()->Union(Lambda_in.Project_1());

  for (auto &psi_in_labda_1 : Theta.Project_1()->Bfs()) {
    auto fiber_labda_0 = Lambda_in.Fiber_0(psi_in_labda_1->node());
    for (auto &psi_in_labda_0 : fiber_labda_0->Bfs())
      for (auto &elem : psi_in_labda_0->node()->support())
        elem->set_marked(true);

    psi_in_labda_1->FrozenOtherAxis()->Union(
        Lambda_out.Project_0(), [](auto &psi_out_labda_0) {
          for (auto &elem : std::get<0>(psi_out_labda_0)->support())
            if (elem->marked()) return true;
          return false;
        });

    for (auto &psi_in_labda_0 : fiber_labda_0->Bfs())
      for (auto &elem : psi_in_labda_0->node()->support())
        elem->set_marked(false);
  }
  return Theta;
}
};  // namespace spacetime
