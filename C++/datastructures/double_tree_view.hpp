#pragma once
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "multi_tree_vector.hpp"
#include "multi_tree_view.hpp"

namespace datastructures {

namespace details {
template <typename I_dbl_node, size_t i>
using T_frozen =
    std::tuple<std::tuple_element_t<i, typename I_dbl_node::TupleNodes>>;
}

template <typename I_dbl_node, size_t i>
class FrozenDoubleNode
    : public MultiNodeViewInterface<FrozenDoubleNode<I_dbl_node, i>,
                                    details::T_frozen<I_dbl_node, i>>,
      public MultiNodeVectorInterface<FrozenDoubleNode<I_dbl_node, i>> {
 private:
  using Super = MultiNodeViewInterface<FrozenDoubleNode<I_dbl_node, i>,
                                       details::T_frozen<I_dbl_node, i>>;
  using Self = FrozenDoubleNode<I_dbl_node, i>;

 public:
  using TupleNodes = details::T_frozen<I_dbl_node, i>;

  explicit FrozenDoubleNode(std::shared_ptr<I_dbl_node> dbl_node)
      : dbl_node_(dbl_node) {
    assert(dbl_node);
  }

  auto nodes() const { return std::tuple{std::get<i>(dbl_node_->nodes())}; }
  auto node() const { return std::get<i>(dbl_node_->nodes()); }
  bool marked() { return dbl_node_->marked(); }
  void set_marked(bool value) { dbl_node_->set_marked(value); }
  auto children(size_t _ = 0) {
    assert(_ == 0);
    std::vector<std::shared_ptr<Self>> result;
    for (auto db_child : dbl_node_->children(i))
      result.push_back(std::make_shared<Self>(db_child));
    return result;
  }
  auto parents(size_t _ = 0) {
    assert(_ == 0);
    std::vector<std::shared_ptr<Self>> result;
    for (auto db_child : dbl_node_->parents(i))
      result.push_back(std::make_shared<Self>(db_child));
    return result;
  }
  auto FrozenOtherAxis() const {
    return std::make_shared<FrozenDoubleNode<I_dbl_node, 1 - i>>(dbl_node_);
  }
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
  std::shared_ptr<I_dbl_node> dbl_node_;
};

template <typename I, typename... T>
class DoubleNodeViewBase : public MultiNodeViewBase<I, T...> {
 private:
  using Super = MultiNodeViewBase<I, T...>;

 public:
  using typename Super::TParents;
  using typename Super::TupleNodes;

  // Constructor for a node.
  explicit DoubleNodeViewBase(const TupleNodes& nodes, const TParents& parents)
      : MultiNodeViewBase<I, T...>(nodes, parents),
        frozen_double_nodes_(
            new FrozenDoubleNode<I, 0>(this->shared_from_this()),
            new FrozenDoubleNode<I, 1>(this->shared_from_this())) {}

  // // Constructor for a root.
  explicit DoubleNodeViewBase(const typename Super::TupleNodes& nodes)
      : DoubleNodeViewBase(nodes, {}) {
    assert(this->is_root());
  }
  explicit DoubleNodeViewBase(T*... nodes)
      : DoubleNodeViewBase(typename Super::TupleNodes(nodes...)) {}

  template <size_t i>
  auto Project() const {
    static_assert(i < 2, "Invalid project");
    return std::get<i>(frozen_double_nodes_);
  }
  auto Project_0() const { return Project<0>(); }
  auto Project_1() const { return Project<1>(); }

 protected:
  std::tuple<std::shared_ptr<FrozenDoubleNode<I, 0>>,
             std::shared_ptr<FrozenDoubleNode<I, 1>>>
      frozen_double_nodes_;
};

// This is is some sincere template hacking
template <typename I, template <typename> typename MT_Base>
class DoubleTreeViewBase : public MT_Base<I> {
 private:
  using Super = MT_Base<I>;

 public:
  using T0p = typename std::tuple_element_t<0, typename I::TupleNodes>;
  using T1p = typename std::tuple_element_t<1, typename I::TupleNodes>;
  using T0 = typename std::remove_pointer_t<T0p>;
  using T1 = typename std::remove_pointer_t<T1p>;
  using MT_Base<I>::MT_Base;

  template <size_t i>
  std::shared_ptr<FrozenDoubleNode<I, i>> Project() const {
    return std::make_shared<FrozenDoubleNode<I, i>>(Super::root);
  }
  std::shared_ptr<FrozenDoubleNode<I, 0>> Fiber_0(T1p mu) const {
    if (!std::get<0>(fibers_).count(mu)) compute_fibers();
    return std::get<0>(fibers_).at(mu);
  }
  std::shared_ptr<FrozenDoubleNode<I, 1>> Fiber_1(T0p mu) const {
    if (!std::get<1>(fibers_).count(mu)) compute_fibers();
    return std::get<1>(fibers_).at(mu);
  }

  // Helper functions..
  auto Project_0() const { return Project<0>(); }
  auto Project_1() const { return Project<1>(); }

  // Set the default parameter for deep copy.
  template <typename MT_other = DoubleTreeViewBase<I, MT_Base>>
  MT_other DeepCopy() const {
    return MT_Base<I>::template DeepCopy<MT_other>();
  }

 protected:
  mutable std::tuple<
      std::unordered_map<T1p, std::shared_ptr<FrozenDoubleNode<I, 0>>>,
      std::unordered_map<T0p, std::shared_ptr<FrozenDoubleNode<I, 1>>>>
      fibers_;

  void compute_fibers() const {
    static_for<2>([&](auto i) {
      for (const auto& f_node :
           Project<i>()->Bfs(/* include_metaroot */ true)) {
        std::get<1 - i>(fibers_).emplace(f_node->node(),
                                         f_node->FrozenOtherAxis());
      }
    });
  }
};

template <typename T0, typename T1>
class DoubleNodeView
    : public DoubleNodeViewBase<DoubleNodeView<T0, T1>, T0, T1> {
 public:
  using DoubleNodeViewBase<DoubleNodeView<T0, T1>, T0, T1>::DoubleNodeViewBase;
};

template <typename T0, typename T1>
using DoubleTreeView =
    DoubleTreeViewBase<DoubleNodeView<T0, T1>, MultiTreeView>;

template <typename T0, typename T1>
using DoubleTreeVector =
    DoubleTreeViewBase<MultiNodeVector<T0, T1>, MultiTreeVector>;

}  // namespace datastructures
