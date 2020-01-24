#pragma once
#include <memory>
#include <tuple>
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
                                    details::T_frozen<I_dbl_node, i>> {
 private:
  using Super = MultiNodeViewInterface<FrozenDoubleNode<I_dbl_node, i>,
                                       details::T_frozen<I_dbl_node, i>>;
  using Self = FrozenDoubleNode<I_dbl_node, i>;

 public:
  explicit FrozenDoubleNode(std::shared_ptr<I_dbl_node> dbl_node)
      : dbl_node_(dbl_node) {}

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

  // Refine is handled by simply refining the underlying double node
  template <size_t _ = 0, typename container, typename Func>
  bool Refine(const container& children_i, const Func& call_filter,
              bool make_conforming) {
    static_assert(_ == 0);
    return dbl_node_->template Refine<i>(children_i, call_filter,
                                         make_conforming);
  }

  // In case this is a vectoral double node.
  inline double value() const { return dbl_node_->value(); }
  inline void set_value(double val) { dbl_node_->set_value(val); }

 protected:
  std::shared_ptr<I_dbl_node> dbl_node_;
};

// This is is some sincere template hacking
template <typename I, template <typename> typename MT_Base>
class DoubleTreeViewBase : public MT_Base<I> {
 private:
  using Super = MT_Base<I>;
  using T0 = typename std::tuple_element_t<0, typename I::TupleNodes>;
  using T1 = typename std::tuple_element_t<1, typename I::TupleNodes>;

 public:
  using MT_Base<I>::MT_Base;

  template <size_t i>
  std::shared_ptr<FrozenDoubleNode<I, i>> Project() {
    return std::make_shared<FrozenDoubleNode<I, i>>(Super::root);
  }
  std::shared_ptr<FrozenDoubleNode<I, 0>> Fiber(T1 mu) {
    if (!std::get<0>(fibers_).count(mu)) compute_fibers();
    return std::get<0>(fibers_).at(mu);
  }
  std::shared_ptr<FrozenDoubleNode<I, 1>> Fiber(T0 mu) {
    if (!std::get<1>(fibers_).count(mu)) compute_fibers();
    return std::get<1>(fibers_).at(mu);
  }

  // Helper functions..
  auto Project_0() { return Project<0>(); }
  auto Project_1() { return Project<1>(); }
  template <size_t i>
  auto Fiber(std::shared_ptr<FrozenDoubleNode<I, i>> mu) {
    return Fiber(mu->node());
  }

 protected:
  std::tuple<std::unordered_map<T1, std::shared_ptr<FrozenDoubleNode<I, 0>>>,
             std::unordered_map<T0, std::shared_ptr<FrozenDoubleNode<I, 1>>>>
      fibers_;

  void compute_fibers() {
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
using DoubleTreeView = DoubleTreeViewBase<MultiNodeView<T0, T1>, MultiTreeView>;

template <typename T0, typename T1>
using DoubleTreeVector =
    DoubleTreeViewBase<MultiNodeVector<T0, T1>, MultiTreeVector>;

}  // namespace datastructures