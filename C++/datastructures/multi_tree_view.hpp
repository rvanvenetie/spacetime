#pragma once
#include <deque>
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
template <typename I, typename... T>
class MultiNodeViewInterface {
 public:
  static constexpr size_t dim = sizeof...(T);
  using TupleNodes = std::tuple<T*...>;

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
  std::vector<I*> Bfs(bool include_metaroot = false,
                      const Func& callback = func_noop,
                      bool return_nodes = true);

  // DeepRefine refines the multitree according to the underlying trees.
  template <typename FuncFilt = T_func_true, typename FuncPost = T_func_noop>
  void DeepRefine(const FuncFilt& call_filter = func_true,
                  const FuncPost& call_postprocess = func_noop);

  template <typename I_other = I, typename FuncFilt = T_func_true,
            typename FuncPost = T_func_noop>
  void Union(I_other* other, const FuncFilt& call_filter = func_true,
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
  friend std::ostream& operator<<(std::ostream& os,
                                  const MultiNodeViewInterface<I, T...>& mnv) {
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
  I* FindBrother(const TupleNodes& nodes, bool make_conforming);
};

template <typename I, typename... T>
class MultiNodeViewBase : public MultiNodeViewInterface<I, T...> {
 public:
  using MultiNodeViewInterface<I, T...>::dim;
  using typename MultiNodeViewInterface<I, T...>::TupleNodes;
  using TParents =
      std::array<StaticVector<I*, std::max({T::N_parents...})>, dim>;
  using TChildren =
      std::array<SmallVector<I*, std::max({T::N_children...})>, dim>;

 public:
  // Constructor for a node.
  explicit MultiNodeViewBase(std::deque<I>* container, const TupleNodes& nodes,
                             const TParents& parents)
      : container_(container), nodes_(nodes), parents_(parents) {
    assert(container);
  }

  // Constructors for root.
  explicit MultiNodeViewBase(std::deque<I>* container, const TupleNodes& nodes)
      : MultiNodeViewBase(container, nodes, {}) {}
  explicit MultiNodeViewBase(std::deque<I>* container, T*... nodes)
      : MultiNodeViewBase(container, TupleNodes(nodes...)) {
    assert(this->is_root());
  }

  // Remove copy constructor.
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

  // Create the ability to make a child.
  inline I* make_child(const TupleNodes& nodes, const TParents& parents) {
    container_->emplace_back(container_, nodes, parents);
    return &container_->back();
  }

  // Access to the deque.
  std::deque<I>* container() { return container_; }

  // Use SFINAE to add convenient functions in case dim == 1.
  template <size_t dim = dim, typename = typename std::enable_if_t<dim == 1>>
  inline auto node() const {
    return std::get<0>(nodes_);
  }
  template <size_t dim = dim, typename = typename std::enable_if_t<dim == 1>>
  inline const auto& parents() const {
    return parents_[0];
  }

 protected:
  bool marked_ = false;
  TupleNodes nodes_;

  // Store parents/children as raw pointers.
  TParents parents_;
  TChildren children_;

  // Pointer to the deque that holds all the childen.
  std::deque<I>* container_;
};

template <typename... T>
class MultiNodeView : public MultiNodeViewBase<MultiNodeView<T...>, T...> {
 public:
  using MultiNodeViewBase<MultiNodeView<T...>, T...>::MultiNodeViewBase;
};

template <typename I>
class MultiTreeView {
 public:
  using Impl = I;
  static constexpr size_t dim = I::dim;

  I* root() { return root_; }
  I* root() const { return root_; }

  // This constructs the tree with a single meta_root.
  template <typename... T>
  explicit MultiTreeView(T... nodes) {
    multi_nodes_.emplace_back(&multi_nodes_, nodes...);
    root_ = &multi_nodes_.back();
    assert(root_->is_root());
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
  std::vector<I*> Bfs(bool include_metaroot = false) const {
    return root_->Bfs(include_metaroot);
  }
  template <typename FuncFilt = T_func_true, typename FuncPost = T_func_noop>
  void DeepRefine(const FuncFilt& call_filter = func_true,
                  const FuncPost& call_postprocess = func_noop) {
    return root_->DeepRefine(call_filter, call_postprocess);
  }
  template <typename T_other = I, typename FuncFilt = T_func_true,
            typename FuncPost = T_func_noop>
  void Union(const T_other& other, const FuncFilt& call_filter = func_true,
             const FuncPost& call_postprocess = func_noop) {
    root_->Union(other.root(), call_filter, call_postprocess);
  }

 protected:
  // Store the root.
  I* root_;

  std::deque<I> multi_nodes_;
};

template <typename T0>
using TreeView = MultiTreeView<MultiNodeView<T0>>;

template <typename T0, typename T1, typename T2>
using TripleTreeView = MultiTreeView<MultiNodeView<T0, T1, T2>>;

};  // namespace datastructures

#include "multi_tree_view.ipp"
