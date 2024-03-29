#pragma once
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "multi_tree_vector.hpp"
#include "multi_tree_view.hpp"

namespace datastructures {

template <typename I_dbl, size_t i>
class FrozenDoubleNode
    : public MultiNodeViewInterface<
          FrozenDoubleNode<I_dbl, i>,
          std::tuple_element_t<i, typename I_dbl::Types>>,
      public MultiNodeVectorInterface<FrozenDoubleNode<I_dbl, i>> {
 private:
  // Some aliases to make life easier.
  using Super =
      MultiNodeViewInterface<FrozenDoubleNode<I_dbl, i>,
                             std::tuple_element_t<i, typename I_dbl::Types>>;
  using Self = FrozenDoubleNode<I_dbl, i>;

 public:
  explicit FrozenDoubleNode(I_dbl* dbl_node) : dbl_node_(dbl_node) {
    assert(dbl_node);
  }

  auto nodes() const { return std::tuple{std::get<i>(dbl_node_->nodes())}; }
  auto node() const { return std::get<i>(dbl_node_->nodes()); }
  bool marked() const { return marked_; }
  void set_marked(bool value) { marked_ = value; }
  const std::vector<Self*>& children(size_t _ = 0) {
    assert(_ == 0);
    const auto& dbl_children = dbl_node_->children(i);
    if (children_.size() != dbl_children.size()) {
      assert(children_.size() < dbl_children.size());
      children_.clear();
      children_.reserve(dbl_children.size());
      for (const auto& db_child : dbl_children)
        children_.emplace_back(db_child->template Frozen<i>());
    }
    return children_;
  }
  auto parents(size_t _ = 0) {
    assert(_ == 0);
    std::vector<Self*> result;
    const auto& dbl_parents = dbl_node_->parents(i);
    result.reserve(dbl_parents.size());
    for (const auto& db_parent : dbl_parents)
      result.emplace_back(db_parent->template Frozen<i>());
    return result;
  }
  auto FrozenOtherAxis() const { return dbl_node_->template Frozen<1 - i>(); }
  auto DoubleNode() const { return dbl_node_; }

  // Refine is handled by simply refining the underlying double node
  template <size_t _ = 0, typename container, typename Func>
  bool Refine(const container& children_i, const Func& call_filter,
              bool make_conforming) {
    static_assert(_ == 0);
    return dbl_node_->template Refine<i>(children_i, call_filter,
                                         make_conforming);
  }
  // Define a practical overload:
  template <size_t _ = 0, typename Func>
  bool Refine(const Func& call_filter, bool make_conforming = false) {
    static_assert(_ == 0);
    return Refine(node()->children(), call_filter, make_conforming);
  }

  // Deep-refines in the `i`-axis.
  template <typename FuncFilt = T_func_true, typename FuncPost = T_func_noop>
  void DeepRefine(const FuncFilt& call_filter = func_true,
                  const FuncPost& call_postprocess = func_noop) {
    this->Bfs(true, [&call_filter, &call_postprocess](const auto& node) {
      call_postprocess(node);
      node->template Refine<i>(
          [&call_filter](auto& nodes) {
            return call_filter(std::get<i>(nodes));
          },
          true);
    });
  }

  // In case this is a vectoral double node.
  inline const double& value() const { return dbl_node_->value(); }
  inline void set_value(double val) { dbl_node_->set_value(val); }

 protected:
  bool marked_ = false;
  I_dbl* dbl_node_;
  std::vector<Self*> children_;
};

// Add the FrozenDoubleNode as member to some MultiNode*.
template <template <typename, typename...> class Base, typename... T>
class DoubleNodeBase : public Base<DoubleNodeBase<Base, T...>, T...> {
 private:
  using Super = Base<DoubleNodeBase<Base, T...>, T...>;
  using I = DoubleNodeBase<Base, T...>;

 public:
  // Some convenient aliases
  using typename Super::TParents;
  using typename Super::TupleNodes;
  using Types = std::tuple<T...>;
  using T0 = std::tuple_element_t<0, Types>;
  using T1 = std::tuple_element_t<1, Types>;

  // Constructor for a node.
  explicit DoubleNodeBase(Deque<I>* container, const TupleNodes& nodes,
                          const TParents& parents)
      : Super(container, nodes, parents), frozen_double_nodes_(this, this) {}

  // // Constructor for a root.
  explicit DoubleNodeBase(Deque<I>* container,
                          const typename Super::TupleNodes& nodes)
      : DoubleNodeBase(container, nodes, {}) {
    assert(this->is_root());
  }
  explicit DoubleNodeBase(Deque<I>* container, T*... nodes)
      : DoubleNodeBase(container, typename Super::TupleNodes(nodes...)) {}

  template <size_t i>
  FrozenDoubleNode<I, i>* Frozen() {
    static_assert(i < 2, "Invalid project");
    return &std::get<i>(frozen_double_nodes_);
  }

  // Some convenient helper functions.
  T0* node_0() const { return std::get<0>(Super::nodes_); }
  T1* node_1() const { return std::get<1>(Super::nodes_); }

 protected:
  std::tuple<FrozenDoubleNode<I, 0>, FrozenDoubleNode<I, 1>>
      frozen_double_nodes_;
};

// This is is some sincere template hacking
template <typename I, template <typename> typename MT_Base>
class DoubleTreeBase : public MT_Base<I> {
 private:
  using Super = MT_Base<I>;

 public:
  using MT_Base<I>::MT_Base;
  using T0 = typename I::T0;
  using T1 = typename I::T1;
  using FrozenDN0Type = FrozenDoubleNode<I, 0>;
  using FrozenDN1Type = FrozenDoubleNode<I, 1>;
  using DNType = I;

  FrozenDoubleNode<I, 0>* Fiber_0(T1* mu) const {
    return std::get<0>(fibers_).at(mu);
  }
  FrozenDoubleNode<I, 1>* Fiber_1(T0* mu) const {
    return std::get<1>(fibers_).at(mu);
  }

  // Helper functions..
  template <size_t i>
  auto Project() const {
    return this->root()->template Frozen<i>();
  }
  auto Project_0() const { return Project<0>(); }
  auto Project_1() const { return Project<1>(); }

  // Set the default parameter for deep copy.
  template <typename MT_other = DoubleTreeBase<I, MT_Base>>
  MT_other DeepCopy() const {
    return MT_Base<I>::template DeepCopy<MT_other>();
  }

  // Calculates the gradedness, and stores the nodes with max gradedness
  // in the given vector.
  size_t Gradedness(std::vector<I*>* max_gradedness = nullptr) const {
    std::vector<I*> nodes = this->root()->Bfs();
    std::unordered_map<I*, int> gradedness;
    int result = 0;

    for (auto dblnode : nodes) {
      if (dblnode->template is_full<0>()) {
        gradedness.emplace(dblnode, 0);
      } else if (dblnode->levels()[1] == 0) {
        gradedness.emplace(dblnode, 1);
      } else {
        int min_grade = INT_MAX;
        for (auto parent : dblnode->parents(1))
          min_grade = std::min(min_grade, gradedness.at(parent));
        gradedness.emplace(dblnode, 1 + min_grade);
      }
      result = std::max(result, gradedness.at(dblnode));
    }

    if (max_gradedness)
      for (auto dblnode : nodes)
        if (gradedness.at(dblnode) == result)
          max_gradedness->emplace_back(dblnode);

    return result;
  }

  void ComputeFibers() const {
    static_for<2>([&](auto i) {
      for (const auto& f_node :
           Project<i>()->Bfs(/* include_metaroot */ true)) {
        std::get<1 - i>(fibers_).emplace(f_node->node(),
                                         f_node->FrozenOtherAxis());
      }
    });
  }

 protected:
  mutable std::tuple<std::unordered_map<T1*, FrozenDoubleNode<I, 0>*>,
                     std::unordered_map<T0*, FrozenDoubleNode<I, 1>*>>
      fibers_;
};

// DoubleNodeView + DoubleTreeView implementation.
template <typename T0, typename T1>
using DoubleNodeView = DoubleNodeBase<MultiNodeViewBase, T0, T1>;

template <typename T0, typename T1>
using DoubleTreeView = DoubleTreeBase<DoubleNodeView<T0, T1>, MultiTreeView>;

// DoubleNodeVector + DoubleTreeVector implementation.
template <typename T0, typename T1>
using DoubleNodeVector = DoubleNodeBase<MultiNodeVectorBase, T0, T1>;

template <typename T0, typename T1>
using DoubleTreeVector =
    DoubleTreeBase<DoubleNodeVector<T0, T1>, MultiTreeVector>;

}  // namespace datastructures
