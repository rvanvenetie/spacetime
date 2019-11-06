#pragma once
#include <deque>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tree.hpp"

#define BOOST_ALLOCATOR
#ifdef BOOST_ALLOCATOR
#define BOOST_POOL_NO_MT
#include <boost/pool/pool_alloc.hpp>
template <typename T>
using VectorAlloc = std::vector<
    T, boost::fast_pool_allocator<T, boost::default_user_allocator_new_delete,
                                  boost::details::pool::null_mutex, 32, 0>>;
template <typename T>
using DequeAlloc = std::deque<T, boost::fast_pool_allocator<T>>;

#else
template <typename I>
using VectorAlloc = std::vector<I>;
#endif

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
constexpr auto func_noop = [](const auto&... x) {};
constexpr auto func_true = [](const auto&... x) { return true; };

// Returns an array of levels of the underlying nodes.
template <typename Ts, size_t dim = std::tuple_size_v<Ts>>
constexpr std::array<int, dim> levels(const Ts& nodes) {
  std::array<int, dim> result;
  static_for<dim>([&](auto i) { result[i] = std::get<i>(nodes)->level(); });
  return result;
}

// Returns the sum of the levels.
template <typename Ts>
constexpr int level(const Ts& nodes) {
  return std::apply([](const auto&... l) { return (l + ...); }, levels(nodes));
}

// Template arguments are as follows:
// I - The final implementation of this class, i.e. the derived class;
// Ts - The tuple type that represents a node;
// dim - The dimension of this multitree.
template <typename I, typename Ts, size_t dim = std::tuple_size_v<Ts>>
class MultiNodeViewInterface {
 public:
  const I& self() const { return static_cast<const I&>(*this); }
  I& self() { return static_cast<I&>(*this); }

  std::array<int, dim> levels() {
    return datastructures::levels(self().nodes());
  }
  int level() const { return datastructures::level(self().nodes()); }

  template <size_t i>
  bool is_full() const {
    return self().children(i).size() ==
           std::get<i>(self().nodes())->children().size();
  }

  bool is_leaf() const {
    for (size_t i = 0; i < dim; ++i)
      if (!self().children(i).empty()) return false;
    return true;
  }

  // Returns whether any of the axis is a metaroot.
  bool is_metaroot() const {
    return std::apply(
        [](const auto&... n) { return (n->is_metaroot() || ...); },
        self().nodes());
  }

  // Returns if this multinode is the root of the multi-tree. That is, whether
  // all axis hold metaroots.
  bool is_root() const {
    return std::apply(
        [](const auto&... n) { return (n->is_metaroot() && ...); },
        self().nodes());
  }

  // Some convenient debug function.
  friend std::ostream& operator<<(std::ostream& os,
                                  const MultiNodeViewInterface<I, Ts>& bla) {
    static_for<dim>([&os, &bla](auto i) {
      if constexpr (i > 0) {
        os << std::string(" x ");
      }
      os << *std::get<i>(bla.self().nodes());
    });
    return os;
  }
};

template <typename I, typename... T>
class MultiNodeView : public MultiNodeViewInterface<I, std::tuple<T*...>> {
 public:
  using Ts = std::tuple<T*...>;
  using ArrayVectorImpls = std::array<VectorAlloc<I*>, sizeof...(T)>;

 public:
  static constexpr size_t dim = sizeof...(T);

  // Constructor for a root.
  explicit MultiNodeView(const Ts& nodes) : nodes_(nodes) {
    assert(this->is_root());
  }
  explicit MultiNodeView(T*... nodes) : MultiNodeView(Ts(nodes...)) {}

  // Constructor for a node.
  explicit MultiNodeView(const Ts& nodes, const ArrayVectorImpls& parents)
      : nodes_(nodes), parents_(parents) {
    static_for<dim>([&](auto i) {
      children_[i].reserve(std::get<i>(nodes)->children().size());
    });
  }

  const Ts& nodes() const { return nodes_; }
  bool marked() const { return marked_; }
  void set_marked(bool value) { marked_ = value; }
  VectorAlloc<I*>& children(size_t i) {
    assert(i < dim);
    return children_[i];
  }
  const VectorAlloc<I*>& children(size_t i) const {
    assert(i < dim);
    return children_[i];
  }

  const VectorAlloc<I*>& parents(size_t i) const {
    assert(i < dim);
    return parents_[i];
  }

 protected:
  bool marked_ = false;
  Ts nodes_;
  ArrayVectorImpls parents_;
  ArrayVectorImpls children_;
};  // namespace datastructures

template <typename... T>
class MultiNodeViewImpl : public MultiNodeView<MultiNodeViewImpl<T...>, T...> {
 public:
  using MultiNodeView<MultiNodeViewImpl<T...>, T...>::MultiNodeView;
};

// Way to inherit NodeView:
//
// struct CrtpFinal;
// template <typename T, typename I = CrtpFinal>
// class NodeView
//    : public MultiNodeView<std::conditional_t<std::is_same_v<I, CrtpFinal>,
//                                              NodeView<T>, NodeView<T, I>>,
//                                              T

template <typename T>
class NodeView : public MultiNodeView<NodeView<T>, T> {
 private:
  using Super = MultiNodeView<NodeView<T>, T>;

 public:
  using MultiNodeView<NodeView<T>, T>::MultiNodeView;

  inline auto& children(size_t i = 0) { return Super::children(0); }
  inline const auto& children(size_t i = 0) const { return Super::children(0); }
  inline const auto& parents(size_t i = 0) const { return Super::parents(0); }
};

template <typename I>
class MultiTree {
 private:
  using Ts = typename I::Ts;
  static constexpr size_t dim = I::dim;

 public:
  I* root;

  // This constructs the tree with a single meta_root.
  template <typename... T>
  explicit MultiTree(T... nodes) : nodes_() {
    root = emplace_back(nodes...);
    assert(root->is_root());
  }

  // Bfs can be used to retrieve the underlying nodes.
  template <typename Func = decltype(func_noop)>
  VectorAlloc<I*> Bfs(bool include_metaroot = false,
                      const Func& callback = func_noop,
                      bool return_nodes = true);

  // DeepRefine refines the multitree according to the underlying trees.
  template <typename FuncFilt = decltype(func_true),
            typename FuncPost = decltype(func_noop)>
  void DeepRefine(const FuncFilt& call_filter = func_true,
                  const FuncPost& call_postprocess = func_noop);

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

  template <typename I_other = I, typename FuncFilt = decltype(func_true),
            typename FuncPost = decltype(func_noop)>
  void Union(I_other* other, const FuncFilt& call_filter = func_true,
             const FuncPost& call_postprocess = func_noop);

  template <typename MT_other = MultiTree<I>,
            typename FuncFilt = decltype(func_true),
            typename FuncPost = decltype(func_noop)>
  void Union(const MT_other& other, const FuncFilt& call_filter = func_true,
             const FuncPost& call_postprocess = func_noop) {
    Union(other.root);
  }

  template <typename MT_other = MultiTree<I>,
            typename FuncPost = decltype(func_noop)>
  MT_other DeepCopy(const FuncPost& call_postprocess = func_noop);

  template <size_t i,
            typename container = std::vector<std::tuple_element_t<i, Ts>>,
            typename Func = decltype(func_true)>
  const VectorAlloc<I*>& Refine(I* multi_node, const container& children_i,
                                const Func& call_filter = func_true,
                                bool make_conforming = false);

  // Define a practical overload:
  template <size_t i, typename FuncFilt = decltype(func_true)>
  const auto& Refine(I* multi_node, const FuncFilt& call_filter = func_true,
                     bool make_conforming = false) {
    return Refine<i>(multi_node, std::get<i>(multi_node->nodes())->children(),
                     call_filter, make_conforming);
  }
  template <typename FuncFilt = decltype(func_true)>
  void Refine(I* multi_node, const FuncFilt& call_filter = func_true,
              bool make_conforming = false) {
    static_for<dim>(
        [&](auto i) { Refine<i>(multi_node, call_filter, make_conforming); });
  }

 protected:
  std::deque<I> nodes_;

  template <typename... Args>
  inline I* emplace_back(Args&&... args) {
    nodes_.emplace_back(std::forward<Args>(args)...);
    return &nodes_.back();
  }

 private:
  template <size_t i, size_t j>
  I* FindBrother(I* multi_node, const Ts& nodes, bool make_conforming);
};

template <typename T1>
using TreeView = MultiTree<NodeView<T1>>;

template <typename T1, typename T2>
using DoubleTreeView = MultiTree<MultiNodeViewImpl<T1, T2>>;

template <typename T1, typename T2, typename T3>
using TripleTreeView = MultiTree<MultiNodeViewImpl<T1, T2, T3>>;
};  // namespace datastructures

#include "multi_tree_view.ipp"
