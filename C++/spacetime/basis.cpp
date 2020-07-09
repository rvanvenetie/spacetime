#include "basis.hpp"

#include <vector>

namespace spacetime {
using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using Time::OrthonormalWaveletFn;
using Time::ThreePointWaveletFn;

template <template <typename, typename> class DblTreeIn,
          template <typename, typename> class DblTreeOut>
DblTreeOut<OrthonormalWaveletFn, HierarchicalBasisFn> GenerateYDelta(
    const DblTreeIn<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta) {
  using SpaceTreeVector =
      std::vector<typename DblTreeIn<ThreePointWaveletFn,
                                     HierarchicalBasisFn>::FrozenDN1Type *>;
  // Figure out the orthonormal metaroot.
  OrthonormalWaveletFn *ortho_meta_root = std::get<0>(X_delta.root()->nodes())
                                              ->children()
                                              .at(0)
                                              ->support()
                                              .at(0)
                                              ->RefinePsiOrthonormal()[0]
                                              ->parents()
                                              .at(0);
  assert(ortho_meta_root->is_metaroot());

  auto Y_delta = DblTreeOut<OrthonormalWaveletFn, HierarchicalBasisFn>(
      ortho_meta_root, std::get<1>(X_delta.root()->nodes()));

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

DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>
GenerateXDeltaUnderscore(
    const DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta,
    size_t num_repeats) {
  assert(num_repeats > 0);
  if (num_repeats > 1)
    return GenerateXDeltaUnderscore(GenerateXDeltaUnderscore(X_delta),
                                    num_repeats - 1);
  DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>
      X_delta_underscore = X_delta.DeepCopy();

  std::vector<typename decltype(X_delta_underscore)::DNType *> time_leaves,
      space_leaves;
  for (auto &dblnode : X_delta_underscore.container()) {
    auto [time_node, space_node] = dblnode.nodes();
    if (!time_node->is_full()) time_node->Refine();
    if (!space_node->is_full())  // This is DIFFERENT from python.
      space_node->Refine();

    if (!dblnode.template is_full<0>()) time_leaves.push_back(&dblnode);
    if (!dblnode.template is_full<1>()) space_leaves.push_back(&dblnode);
  }

  for (auto dblnode : time_leaves)
    dblnode->template Refine<0>(datastructures::func_true,
                                /*make_conforming*/ true);

  for (auto dblnode : space_leaves) {
    dblnode->template Refine<1>(datastructures::func_true,
                                /*make_conforming*/ true);
    for (auto child : dblnode->children(1)) {
      child->node_1()->Refine();
      child->template Refine<1>(datastructures::func_true,
                                /*make_conforming*/ true);
    }
  }

  return X_delta_underscore;
}

// Template specializations for GenerateYDelta.
template DoubleTreeView<OrthonormalWaveletFn, HierarchicalBasisFn>
GenerateYDelta<DoubleTreeView, DoubleTreeView>(
    const DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta);
template DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>
GenerateYDelta<DoubleTreeVector, DoubleTreeVector>(
    const DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn> &X_delta);

};  // namespace spacetime
