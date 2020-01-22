#pragma once
#include <iostream>

#include "multi_tree_view.hpp"

namespace datastructures {

template <typename I, typename TupleNodes>
template <typename Func>
inline std::vector<I*> MultiNodeViewInterface<I, TupleNodes>::Bfs(
    bool include_metaroot, const Func& callback, bool return_nodes) {
  assert(is_root() && !self().marked());
  std::vector<I*> nodes;
  std::queue<I*> queue;

  queue.emplace(static_cast<I*>(this));
  self().set_marked(true);
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();

    nodes.emplace_back(node);
    callback(node);
    for (size_t i = 0; i < dim; ++i)
      for (const auto& child : node->children(i))
        if (!child->marked()) {
          child->set_marked(true);
          queue.emplace(child.get());
        }
  }
  for (const auto& node : nodes) {
    node->set_marked(false);
  }
  if (!return_nodes) return {};
  if (!include_metaroot) {
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
                       [](const auto& node) { return node->is_metaroot(); }),
        nodes.end());
  }
  return nodes;
}

template <typename I, typename TupleNodes>
template <typename FuncFilt, typename FuncPost>
inline void MultiNodeViewInterface<I, TupleNodes>::DeepRefine(
    const FuncFilt& call_filter, const FuncPost& call_postprocess) {
  // This callback will invoke call_postprocess, and refine according
  // to the given filter.
  auto callback = [&call_filter, &call_postprocess](const auto& node) {
    call_postprocess(node);

    // Simply refine all the children, using static loop.
    node->Refine(/*call_filter*/ call_filter,
                 /*make_conforming*/ true);
  };

  // Now simply invoke Bfs.
  Bfs(/*include_metaroot*/ true, /*callback*/ callback,
      /*return_nodes*/ false);
}

template <typename I, typename TupleNodes>
template <typename I_other, typename FuncFilt, typename FuncPost>
void MultiNodeViewInterface<I, TupleNodes>::Union(
    std::shared_ptr<I_other> other, const FuncFilt& call_filter,
    const FuncPost& call_postprocess) {
  assert(is_root() && other->is_root());
  assert(self().nodes() == other->nodes());
  std::queue<std::pair<I*, I_other*>> queue;
  queue.emplace(static_cast<I*>(this), other.get());
  self().set_marked(true);

  std::vector<I*> my_nodes;
  while (!queue.empty()) {
    I* my_node;
    I_other* other_node;

    std::tie(my_node, other_node) = queue.front();
    queue.pop();

    assert(my_node->nodes() == other_node->nodes());
    call_postprocess(my_node, other_node);
    my_nodes.emplace_back(my_node);

    // Now do the union magic in all dimensions.
    static_for<dim>([&queue, &my_node, &other_node, &call_filter](auto i) {
      // Get a list of all children of the other_node in axis `i`.
      static std::vector<std::tuple_element_t<i, TupleNodes>> other_children_i;
      other_children_i.clear();
      for (const auto& other_child_i : other_node->children(i))
        other_children_i.emplace_back(std::get<i>(other_child_i->nodes()));

      // Refine my_node using this list of children.
      my_node->template Refine<i>(other_children_i, call_filter,
                                  /*make_conforming*/ false);

      // Now only put children on the queue that other_node has as well.
      for (const auto& my_child : my_node->children(i)) {
        if (my_child->marked()) continue;
        for (const auto& other_child : other_node->children(i))
          if (my_child->nodes() == other_child->nodes()) {
            my_child->set_marked(true);
            queue.emplace(my_child.get(), other_child.get());
            break;
          }
      }
    });
  }
  // Reset mark field.
  for (const auto& my_node : my_nodes) {
    my_node->set_marked(false);
  }
}

template <typename I, typename TupleNodes>
template <size_t i, typename container, typename Func>
inline bool MultiNodeViewInterface<I, TupleNodes>::Refine(
    const container& children_i, const Func& call_filter,
    bool make_conforming) {
  static_assert(i < dim);
  if (is_full<i>()) return false;

  const auto& nodes = self().nodes();
  const auto& nodes_i = std::get<i>(nodes);
  bool refined = false;
  for (const auto& child_i : children_i) {
    // Assert that this child lies in the underlying tree.
    assert(std::find(nodes_i->children().begin(), nodes_i->children().end(),
                     child_i) != nodes_i->children().end());

    // Create a copy of nodes, with the i-th index replaced by child_i.
    TupleNodes child_nodes(nodes);
    std::get<i>(child_nodes) = child_i;

    // Skip if this child already exists.
    if (std::any_of(self().children(i).begin(), self().children(i).end(),
                    [&child_nodes](const auto& child) {
                      return child->nodes() == child_nodes;
                    }))
      continue;

    // Skip if the call_filter doesn't pass.
    if constexpr (dim == 1) {
      if (!call_filter(child_i)) continue;
    } else {
      if (!call_filter(child_nodes)) continue;
    }

    // Collect all brothers.
    typename I::TParents brothers;

    // Find brothers in all axes, using a static for loop over j.
    static_for<dim>([make_conforming, &brothers, &child_nodes, this](auto j) {
      for (const auto& child_parent_j : std::get<j>(child_nodes)->parents()) {
        // Create a copy of the child_nodes, replacing j-th index.
        TupleNodes brother_nodes(child_nodes);
        std::get<j>(brother_nodes) = child_parent_j->shared_from_this();
        brothers[j].push_back(
            FindBrother<i, j>(brother_nodes, make_conforming));
      }
    });

    // Now finally, lets create an actual child!
#ifndef BOOST_ALLOCATOR
    auto child = std::make_shared<I>(/*nodes*/ std::move(child_nodes),
                                     /*parents*/ std::move(brothers));
#else
    typedef boost::fast_pool_allocator<I> BoostAlloc;
    auto child = std::allocate_shared<I, BoostAlloc>(
        BoostAlloc(), /*nodes*/ std::move(child_nodes),
        /*parents*/ std::move(brothers));
#endif

    // Add this child to all brothers.
    for (size_t j = 0; j < dim; ++j) {
      for (const auto& brother : child->parents(j)) {
        brother->children(j).push_back(child);
      }
    }
    refined = true;
  }
  return refined;
}

template <typename I, typename TupleNodes>
template <size_t i, size_t j>
inline I* MultiNodeViewInterface<I, TupleNodes>::FindBrother(
    const TupleNodes& nodes, bool make_conforming) {
  if (nodes == self().nodes()) {
    return static_cast<I*>(this);
  }

  for (const auto& parent_j : self().parents(j)) {
    for (const auto& sibling_i : parent_j->children(i)) {
      if (sibling_i->nodes() == nodes) {
        return sibling_i.get();
      }
    }
  }

  // We didn't find the brother, lets create it.
  assert(make_conforming);
  const auto& nodes_i = std::get<i>(nodes);
  for (const auto& parent_j : self().parents(j)) {
    // Check whether parent_j has node_i as child in the underlying tree.
    const auto& nodes_parent_j = std::get<i>(parent_j->nodes());
    if (std::find(nodes_parent_j->children().begin(),
                  nodes_parent_j->children().end(),
                  nodes_i) != nodes_parent_j->children().end()) {
      parent_j->template Refine<i>(std::array{nodes_i},
                                   /*call_filter*/ func_true,
                                   /*make_conforming*/ true);
    }
  }

  // Try calling this function again.
  return FindBrother<i, j>(nodes, /*make_conforming*/ false);
}

template <typename I>
inline void MultiTreeView<I>::UniformRefine(std::array<int, dim> max_levels) {
  DeepRefine([&](const typename I::TupleNodes& nodes) {
    bool result = true;
    static_for<dim>([&](auto i) {
      result = result && (std::get<i>(nodes)->level() <= max_levels[i]);
    });
    return result;
  });
}

template <typename I>
inline void MultiTreeView<I>::SparseRefine(int max_level,
                                           std::array<int, dim> weights) {
  DeepRefine([&](const typename I::TupleNodes& nodes) {
    auto lvls = levels(nodes);
    int w_level = 0;
    for (int i = 0; i < dim; ++i) w_level += weights[i] * lvls[i];
    return w_level <= max_level;
  });
}

template <typename I>
template <typename MT_other, typename FuncPost>
inline MT_other MultiTreeView<I>::DeepCopy(
    const FuncPost& call_postprocess) const {
  assert(root->is_root());
  MT_other new_tree(root->nodes());
  new_tree.root->Union(root, /*call_filter*/ func_true, call_postprocess);
  return new_tree;
}
};  // namespace datastructures
