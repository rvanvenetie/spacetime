#pragma once
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "boost.hpp"
#include "tree.hpp"

namespace datastructures {

// Below are two helper functions to create a compile-time-unrolled loop.
// It is implemented using operator folding for the comma operator.
template <typename Func, size_t... Is>
constexpr void static_for_impl(Func&& f, std::index_sequence<Is...>) {
  (f(std::integral_constant<size_t, Is>{}), ...);
}

template <size_t N, typename Func>
constexpr void static_for(Func&& f) {
  static_for_impl(std::forward<Func>(f), std::make_index_sequence<N>{});
}

// Returns an array of levels of the underlying nodes.
template <typename TupleNodes, size_t dim = std::tuple_size_v<TupleNodes>>
constexpr std::array<int, dim> levels(const TupleNodes& nodes) {
  std::array<int, dim> result{-1337};
  static_for<dim>([&](auto i) { result[i] = std::get<i>(nodes)->level(); });
  return result;
}

// Returns the sum of the levels.
template <typename TupleNodes>
constexpr int level(const TupleNodes& nodes) {
  return std::apply([](const auto&... l) { return (l + ...); }, levels(nodes));
}

// Template arguments are as follows:
// I - The final implementation of this class, i.e. the derived class;
// TupleNodes - The tuple type that represents a node;
template <typename I, typename TupleNodes>
class MultiNodeViewInterface : public std::enable_shared_from_this<I> {
 public:
  constexpr static size_t dim = std::tuple_size_v<TupleNodes>;

  inline const I& self() const { return static_cast<const I&>(*this); }
  inline I& self() { return static_cast<I&>(*this); }

  std::array<int, dim> levels() {
    return datastructures::levels(self().nodes());
  }
  int level() const { return datastructures::level(self().nodes()); }

  template <size_t i>
  inline bool is_full() const {
    return self().children(i).size() ==
           std::get<i>(self().nodes())->children().size();
  }

  bool is_leaf() const {
    for (size_t i = 0; i < dim; ++i)
      if (!self().children(i).empty()) return false;
    return true;
  }

  // Returns whether any of the axis is a metaroot.
  inline bool is_metaroot() const {
    return std::apply(
        [](const auto&... n) { return (n->is_metaroot() || ...); },
        self().nodes());
  }

  // Returns if this multinode is the root of the multi-tree. That is, whether
  // all axis hold metaroots.
  inline bool is_root() const {
    return std::apply(
        [](const auto&... n) { return (n->is_metaroot() && ...); },
        self().nodes());
  }

  // Bfs can be used to retrieve the underlying nodes.
  template <typename Func = T_func_noop>
  std::vector<std::shared_ptr<I>> Bfs(bool include_metaroot = false,
                                      const Func& callback = func_noop,
                                      bool return_nodes = true);

  // DeepRefine refines the multitree according to the underlying trees.
  template <typename FuncFilt = T_func_true, typename FuncPost = T_func_noop>
  void DeepRefine(const FuncFilt& call_filter = func_true,
                  const FuncPost& call_postprocess = func_noop);

  template <typename I_other = I, typename FuncFilt = T_func_true,
            typename FuncPost = T_func_noop>
  void Union(std::shared_ptr<I_other> other,
             const FuncFilt& call_filter = func_true,
             const FuncPost& call_postprocess = func_noop);

  template <
      size_t i,
      typename container = std::vector<std::tuple_element_t<i, TupleNodes>>,
      typename Func = T_func_true>
  // Returns whether we actually created new nodes.
  bool Refine(const container& children_i, const Func& call_filter = func_true,
              bool make_conforming = false);

  // Define a practical overload:
  template <size_t i, typename FuncFilt = T_func_true>
  bool Refine(const FuncFilt& call_filter = func_true,
              bool make_conforming = false) {
    return Refine<i>(std::get<i>(self().nodes())->children(), call_filter,
                     make_conforming);
  }
  template <typename FuncFilt = T_func_true>
  bool Refine(const FuncFilt& call_filter = func_true,
              bool make_conforming = false) {
    bool result = false;
    static_for<dim>(
        [&](auto i) { result |= Refine<i>(call_filter, make_conforming); });
    return result;
  }

  // Some convenient debug function.
  friend std::ostream& operator<<(
      std::ostream& os, const MultiNodeViewInterface<I, TupleNodes>& mnv) {
    static_for<dim>([&os, &mnv](auto i) {
      if constexpr (i > 0) {
        os << std::string(" x ");
      }
      os << *std::get<i>(mnv.self().nodes());
    });
    return os;
  }

 private:
  template <size_t i, size_t j>
  std::shared_ptr<I> FindBrother(const TupleNodes& nodes, bool make_conforming);
};

template <typename I, typename... T>
class MultiNodeViewBase : public MultiNodeViewInterface<I, std::tuple<T*...>> {
 public:
  static constexpr size_t dim = sizeof...(T);
  using TupleNodes = std::tuple<T*...>;
  using TParents =
      std::array<StaticVector<I*, std::max({T::N_parents...})>, dim>;
  using TChildren =
      std::array<SmallVector<std::shared_ptr<I>, std::max({T::N_children...})>,
                 dim>;

 public:
  // Constructor for a root.
  explicit MultiNodeViewBase(const TupleNodes& nodes) : nodes_(nodes) {
    assert(this->is_root());
  }
  explicit MultiNodeViewBase(T*... nodes)
      : MultiNodeViewBase(TupleNodes(nodes...)) {}

  // Constructor for a node.
  explicit MultiNodeViewBase(TupleNodes&& nodes, TParents&& parents)
      : nodes_(std::move(nodes)), parents_(std::move(parents)) {}

  MultiNodeViewBase(const MultiNodeViewBase&) = delete;

  const TupleNodes& nodes() const { return nodes_; }
  bool marked() const { return marked_; }
  void set_marked(bool value) { marked_ = value; }
  auto& children(size_t i) {
    assert(i < dim);
    return children_[i];
  }
  const auto& children(size_t i) const {
    assert(i < dim);
    return children_[i];
  }

  const auto& parents(size_t i) const {
    assert(i < dim);
    return parents_[i];
  }

 protected:
  bool marked_ = false;
  TupleNodes nodes_;
  TParents parents_;
  TChildren children_;
};

template <typename... T>
class MultiNodeView : public MultiNodeViewBase<MultiNodeView<T...>, T...> {
 public:
  using MultiNodeViewBase<MultiNodeView<T...>, T...>::MultiNodeViewBase;
};

// Way to inherit NodeView:
//
// struct CrtpFinal;
// template <typename T, typename I = CrtpFinal>
// class NodeView
//    : public MultiNodeView<std::conditional_t<std::is_same_v<I, CrtpFinal>,
//                                              NodeView<T>, NodeView<T, I>>,
//                                              T

template <typename I, typename T>
class NodeViewBase : public MultiNodeViewBase<I, T> {
 private:
  using Super = MultiNodeViewBase<I, T>;

 public:
  using MultiNodeViewBase<I, T>::MultiNodeViewBase;

  inline auto& children(size_t i = 0) { return Super::children(0); }
  inline const auto& children(size_t i = 0) const { return Super::children(0); }
  inline const auto& parents(size_t i = 0) const { return Super::parents(0); }
  inline T* node() const { return std::get<0>(Super::nodes_); }
};

template <typename T>
class NodeView : public NodeViewBase<NodeView<T>, T> {
 public:
  using NodeViewBase<NodeView<T>, T>::NodeViewBase;
};

template <typename I>
class MultiTreeView {
 public:
  using Impl = I;
  static constexpr size_t dim = I::dim;
  std::shared_ptr<I> root;

  // This constructs the tree with a single meta_root.
  template <typename... T>
  explicit MultiTreeView(T... nodes) {
    root = std::make_shared<I>(nodes...);
    assert(root->is_root());
  }

  template <typename... T>
  explicit MultiTreeView(const std::unique_ptr<T>&... nodes)
      : MultiTreeView(nodes.get()...) {}

  MultiTreeView(const MultiTreeView<I>&) = delete;
  MultiTreeView(MultiTreeView<I>&&) = default;

  // Uniform refine, nodes->level() <= max_levels.
  void UniformRefine(std::array<int, dim> max_levels);
  void UniformRefine(int max_level) {
    std::array<int, dim> arg;
    arg.fill(max_level);
    UniformRefine(arg);
  }

  // Sparse refine, lin_comb(nodes->level()) <= max_level
  void SparseRefine(int max_level, std::array<int, dim> weights);
  void SparseRefine(int max_level) {
    std::array<int, dim> arg;
    arg.fill(1);
    SparseRefine(max_level, arg);
  }

  template <typename MT_other = MultiTreeView<I>,
            typename FuncPost = T_func_noop>
  MT_other DeepCopy(const FuncPost& call_postprocess = func_noop) const;

  // Simple helpers.
  std::vector<std::shared_ptr<I>> Bfs(bool include_metaroot = false) const {
    return root->Bfs(include_metaroot);
  }
  template <typename FuncFilt = T_func_true, typename FuncPost = T_func_noop>
  void DeepRefine(const FuncFilt& call_filter = func_true,
                  const FuncPost& call_postprocess = func_noop) {
    return root->DeepRefine(call_filter, call_postprocess);
  }
};

template <typename T0>
using TreeView = MultiTreeView<NodeView<T0>>;

template <typename T0, typename T1, typename T2>
using TripleTreeView = MultiTreeView<MultiNodeView<T0, T1, T2>>;

};  // namespace datastructures

#include "multi_tree_view.ipp"
