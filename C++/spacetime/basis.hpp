#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"
#include "../time/haar_basis.hpp"
#include "../time/orthonormal_basis.hpp"
#include "../time/three_point_basis.hpp"

namespace spacetime {
datastructures::DoubleTreeView<Time::OrthonormalWaveletFn,
                               space::HierarchicalBasisFn>
GenerateYDelta(const datastructures::DoubleTreeView<
               Time::ThreePointWaveletFn, space::HierarchicalBasisFn> &X_delta);

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
};  // namespace spacetime
