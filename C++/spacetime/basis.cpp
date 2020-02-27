#include "basis.hpp"

#include <vector>

using datastructures::DoubleNodeView;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::ortho_tree;
using Time::OrthonormalWaveletFn;
using Time::three_point_tree;
using Time::ThreePointWaveletFn;
using SpaceTreeVector = std::vector<datastructures::FrozenDoubleNode<
    datastructures::DoubleNodeView<ThreePointWaveletFn, HierarchicalBasisFn>,
    /*i*/ 1> *>;

namespace spacetime {
DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn> GenerateYDelta(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta) {
  auto Y_delta = DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>(
      ortho_tree.meta_root.get(), std::get<1>(X_delta.root()->nodes()));

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

DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>
GenerateXDeltaUnderscore(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta) {
  auto X_delta_underscore = X_delta.DeepCopy();
  std::vector<DoubleNodeView<ThreePointWaveletFn, HierarchicalBasisFn> *>
      time_leaves, space_leaves;
  for (auto dblnode : X_delta_underscore.Bfs()) {
    auto [time_node, space_node] = dblnode->nodes();
    if (!time_node->is_full()) time_node->Refine();
    if (!space_node->is_full())  // This is DIFFERENT from python.
      space_node->Refine();

    if (!dblnode->is_full<0>()) time_leaves.push_back(dblnode);
    if (!dblnode->is_full<1>()) space_leaves.push_back(dblnode);
  }

  for (auto dblnode : time_leaves) {
    std::cout << "time refine" << std::endl;
    dblnode->Refine<0>(datastructures::func_true, /*make_conforming*/ true);
    std::cout << "end time refine" << std::endl;
  }

  for (auto dblnode : space_leaves) {
    std::cout << "space refine" << std::endl;
    dblnode->Refine<1>(datastructures::func_true, /*make_conforming*/ true);
    std::cout << "end space refine" << std::endl;
    for (auto child : dblnode->children(1)) {
      child->node_1()->Refine();
      std::cout << "child refine" << std::endl;
      child->Refine<1>(datastructures::func_true, /*make_conforming*/ true);
      std::cout << "end child refine" << std::endl;
    }
  }

  return X_delta_underscore;
}
};  // namespace spacetime
