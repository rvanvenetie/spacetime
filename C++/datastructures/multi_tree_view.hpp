#pragma once
#include <experimental/tuple>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>
#include "tree.hpp"

namespace datastructures {

template <typename Func, std::size_t... Is>
constexpr void static_for_impl(Func&& f, std::index_sequence<Is...>) {
  (f(std::integral_constant<std::size_t, Is>{}), ...);
}

template <size_t N, typename Func>
constexpr void static_for(Func&& f) {
  static_for_impl(std::forward<Func>(f), std::make_index_sequence<N>{});
}

template <typename I, size_t dim>
class MultiNodeViewInterface {
 public:
  int level() const {
    const auto& nodes = static_cast<const I*>(this)->nodes();
    // size_t result = 0;
    // static_for<dim>([&result, &nodes](auto index) {
    //  result += std::get<index>(nodes)->level();
    //});
    // return result;

    return std::experimental::apply(
        [](auto const&... e) -> decltype(auto) { return (e->level() + ...); },
        nodes);
  }

  template <size_t i>
  bool IsFull() const {
    auto self = static_cast<const I*>(this);
    return self->children(i).size() ==
           std::get<i>(self->nodes())->children().size();
  }

  template <typename T>
  std::shared_ptr<I> FindBrother(const T& nodes, size_t j, size_t i) {
    auto self = static_cast<I*>(this);
    if (nodes == self->nodes()) {
      return self->shared_from_this();
    }

    for (const auto& parent_j : self->parents(j)) {
      for (const auto& sibling_i : parent_j->children(i)) {
        if (sibling_i->nodes() == nodes) {
          return sibling_i;
        }
      }
    }
    assert(false);
  }

  template <size_t j, typename T>
  void FindBrothers(const T& child_nodes, size_t i,
                    std::array<std::vector<std::shared_ptr<I>>, dim>& result) {
    for (auto child_parent_j : std::get<j>(child_nodes)->parents()) {
      auto brother_nodes{child_nodes};  // Create a copy
      std::get<j>(brother_nodes) = child_parent_j;
      result[j].push_back(FindBrother(brother_nodes, j, i));
    }

    if constexpr (j > 0) {
      FindBrothers<j - 1>(child_nodes, i, result);
    }
  }

  template <typename T>
  std::array<std::vector<std::shared_ptr<I>>, dim> FindBrothers(
      const T& child_nodes, size_t i) {
    std::array<std::vector<std::shared_ptr<I>>, dim> result;

    FindBrothers<dim - 1>(child_nodes, i, result);
    return result;
  }

  template <size_t i, typename T>
  const auto& Refine(const std::vector<std::shared_ptr<T>>& children) {
    auto self = static_cast<I*>(this);
    if (IsFull<i>()) return self->children(i);

    auto nodes = self->nodes();
    auto nodes_i = std::get<i>(nodes);
    for (const auto& child_i : children) {
      if (std::find(nodes_i->children().begin(), nodes_i->children().end(),
                    child_i) == nodes_i->children().end())
        continue;

      auto child_nodes{nodes};  // Create a copy
      std::get<i>(child_nodes) = child_i;

      // Skip if this child already exists.
      if (std::find_if(self->children(i).begin(), self->children(i).end(),
                       [child_nodes](auto child) {
                         return child->nodes() == child_nodes;
                       }) != self->children(i).end())
        continue;

      // Find brothers.
      auto brothers = FindBrothers(child_nodes, i);
    }
    /*
            for child_i in children:
    #If this child does not exist in underlying tree, we can stop.
                if child_i not in self.nodes[i].children: continue

                child_nodes = _replace(i, self.nodes, child_i)

    #Skip if this child already exists, or if the filter doesn't pass.
                if child_nodes in (n.nodes for n in self._children[i]) \
                        or not call_filter(child_nodes):
                    continue

    #Ensure all the parents of the to be created child exist.
    #These are brothers in various axis of the current node.
                brothers = []
                for j in range(self.dim):
                    brothers.append([
                        self._find_brother(
                            _replace(j, child_nodes, child_parent_j), j, i,
                            make_conforming)
                        for child_parent_j in child_nodes[j].parents
                    ])

                child = self.__class__(nodes=child_nodes, parents=brothers)
                for j in range(self.dim):
                    for brother in brothers[j]:
                        brother._children[j].append(child)

            return self._children[i]
    */
  }

  template <size_t i>
  const auto& Refine() {
    return Refine<i>(std::get<i>(static_cast<I*>(this)->nodes())->children());
  }
};

template <typename I, typename... T>
class MultiNodeView : public MultiNodeViewInterface<I, sizeof...(T)> {
 public:
  using TupleNodes = std::tuple<std::shared_ptr<T...>>;
  using ArrayVectorImpls =
      std::array<std::vector<std::shared_ptr<I>>, sizeof...(T)>;

 public:
  static const size_t dim = sizeof...(T);

  explicit MultiNodeView(const TupleNodes& nodes,
                         const ArrayVectorImpls& parents)
      : nodes_(nodes), parents_(parents) {}

  explicit MultiNodeView(const TupleNodes& nodes) : nodes_(nodes) {}
  explicit MultiNodeView(std::shared_ptr<T...> nodes)
      : nodes_(std::make_tuple(nodes)) {}

  const TupleNodes& nodes() const { return nodes_; }
  inline std::vector<std::shared_ptr<I>>& children(size_t i) {
    return children_[i];
  }
  inline const std::vector<std::shared_ptr<I>>& children(size_t i) const {
    return children_[i];
  }

  inline const std::vector<std::shared_ptr<I>>& parents(size_t i) const {
    return parents_[i];
  }

 protected:
  bool marked_ = false;
  TupleNodes nodes_;
  ArrayVectorImpls parents_;
  ArrayVectorImpls children_;
};

template <typename T>
class NodeView : public MultiNodeView<NodeView<T>, T>,
                 public NodeInterface<NodeView<T>> {
 public:
  explicit NodeView(std::shared_ptr<T> node)
      : MultiNodeView<NodeView<T>, T>(node) {}

 protected:
  using MultiNodeView<NodeView<T>, T>::parents_;
  using MultiNodeView<NodeView<T>, T>::children_;
};
};  // namespace datastructures
