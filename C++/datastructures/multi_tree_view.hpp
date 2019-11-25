#pragma once
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
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
#endif

template <typename I, size_t N>
using SmallVector = boost::container::small_vector<I, N>;

template <typename I, size_t N>
using StaticVector = boost::container::static_vector<I, N>;

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

// Template arguments are as follows:
// I - The final implementation of this class, i.e. the derived class;
// Ts - The tuple type that represents a node;
// dim - The dimension of this multitree.
template <typename I, typename Ts, size_t dim = std::tuple_size_v<Ts>>
class MultiNodeViewInterface : public std::enable_shared_from_this<I> {
 public:
  static constexpr auto func_noop = [](const auto&... x) {};
  static constexpr auto func_true = [](const auto&... x) { return true; };

  const I& self() const { return static_cast<const I&>(*this); }
  I& self() { return static_cast<I&>(*this); }

  // Returns an array of levels of the underlying nodes.
  static constexpr std::array<int, dim> levels(const Ts& nodes) {
    std::array<int, dim> result;
    static_for<dim>([&](auto i) { result[i] = std::get<i>(nodes)->level(); });
    return result;
  }
  std::array<int, dim> levels() const { return levels(self().nodes()); }

  // Returns the sum of the levels.
  static constexpr int level(const Ts& nodes) {
    return std::apply([](const auto&... l) { return (l + ...); },
                      levels(nodes));
  }
  int level() const { return level(self().nodes()); }

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

  // Bfs can be used to retrieve the underlying nodes.
  template <typename Func = decltype(func_noop)>
  std::vector<std::shared_ptr<I>> Bfs(bool include_metaroot = false,
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
  void Union(std::shared_ptr<I_other> other,
             const FuncFilt& call_filter = func_true,
             const FuncPost& call_postprocess = func_noop);

  template <typename I_other = I, typename FuncPost = decltype(func_noop)>
  std::shared_ptr<I_other> DeepCopy(
      const FuncPost& call_postprocess = func_noop);

  template <size_t i,
            typename container = std::vector<std::tuple_element_t<i, Ts>>,
            typename Func = decltype(func_true)>
  const auto& Refine(const container& children_i,
                     const Func& call_filter = func_true,
                     bool make_conforming = false);

  // Define a practical overload:
  template <size_t i, typename FuncFilt = decltype(func_true)>
  const auto& Refine(const FuncFilt& call_filter = func_true,
                     bool make_conforming = false) {
    return Refine<i>(std::get<i>(self().nodes())->children(), call_filter,
                     make_conforming);
  }
  template <typename FuncFilt = decltype(func_true)>
  void Refine(const FuncFilt& call_filter = func_true,
              bool make_conforming = false) {
    static_for<dim>([&](auto i) { Refine<i>(call_filter, make_conforming); });
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

 private:
  template <size_t i, size_t j>
  std::shared_ptr<I> FindBrother(const Ts& nodes, bool make_conforming);
};

template <typename I, typename... T>
class MultiNodeView
    : public MultiNodeViewInterface<I, std::tuple<std::shared_ptr<T>...>> {
 public:
  static constexpr size_t dim = sizeof...(T);
  using TupleNodes = std::tuple<std::shared_ptr<T>...>;
  using TParents = std::array<StaticVector<I*, 2>, dim>;
  using TChildren = std::array<SmallVector<std::shared_ptr<I>, 4>, dim>;

 public:
  explicit MultiNodeView(TupleNodes&& nodes, TParents&& parents)
      : nodes_(std::move(nodes)), parents_(std::move(parents)) {
    //    static_for<dim>([&](auto i) {
    //      children_[i].reserve(std::get<i>(nodes_)->children().size());
    //    });
  }
  MultiNodeView() {}
  static std::shared_ptr<I> CreateRoot(std::shared_ptr<T>... nodes) {
    auto result = std::make_shared<I>();
    result->nodes_ = std::make_tuple(nodes...);
    assert(result->is_root());
    return result;
  }

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

// Way to inherit NodeView:
//
// struct CrtpFinal;
// template <typename T, typename I = CrtpFinal>
// class NodeView
//    : public MultiNodeView<std::conditional_t<std::is_same_v<I, CrtpFinal>,
//                                              NodeView<T>, NodeView<T, I>>,
//                                              T

template <typename I, typename T>
class NodeViewBase : public MultiNodeView<I, T> {
 private:
  using Super = MultiNodeView<I, T>;

 public:
  using MultiNodeView<I, T>::MultiNodeView;

  inline auto& children(size_t i = 0) { return Super::children(0); }
  inline const auto& children(size_t i = 0) const { return Super::children(0); }
  inline const auto& parents(size_t i = 0) const { return Super::parents(0); }
  inline std::shared_ptr<T> node() const { return std::get<0>(Super::nodes_); }
};

template <typename T>
class NodeView : public NodeViewBase<NodeView<T>, T> {
 public:
  using NodeViewBase<NodeView<T>, T>::NodeViewBase;
};

template <typename T1, typename T2>
class DoubleNodeView : public MultiNodeView<DoubleNodeView<T1, T2>, T1, T2> {
  using MultiNodeView<DoubleNodeView<T1, T2>, T1, T2>::MultiNodeView;
};

template <typename T1, typename T2, typename T3>
class TripleNodeView
    : public MultiNodeView<TripleNodeView<T1, T2, T3>, T1, T2, T3> {
 public:
  using MultiNodeView<TripleNodeView<T1, T2, T3>, T1, T2, T3>::MultiNodeView;
};
};  // namespace datastructures

#include "multi_tree_view.ipp"
