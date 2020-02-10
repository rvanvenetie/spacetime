#include "basis.hpp"

#include <vector>

using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::ortho_tree;
using Time::OrthonormalWaveletFn;
using Time::three_point_tree;
using Time::ThreePointWaveletFn;
using SpaceTreeVector =
    std::vector<std::shared_ptr<datastructures::FrozenDoubleNode<
        datastructures::DoubleNodeView<ThreePointWaveletFn,
                                       HierarchicalBasisFn>,
        /*i*/ 1>>>;

namespace spacetime {
DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn> GenerateYDelta(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta) {
  auto Y_delta = DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>(
      ortho_tree.meta_root.get(), std::get<1>(X_delta.root->nodes()));

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
};  // namespace spacetime
