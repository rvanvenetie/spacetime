#pragma once
#include <iostream>

#include "datastructures/multi_tree_view.hpp"

namespace datastructures {

template <typename I, typename Ts, size_t dim>
template <typename Func>
inline std::vector<std::shared_ptr<I>> MultiNodeViewInterface<I, Ts, dim>::Bfs(
    bool include_metaroot, const Func& callback, bool return_nodes) {
  assert(is_root());
  std::vector<std::shared_ptr<I>> nodes;
  std::queue<std::shared_ptr<I>> queue;
  queue.emplace(self().shared_from_this());
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();
    if (node->marked()) continue;
    nodes.emplace_back(node);
    node->set_marked(true);
    callback(node);
    for (size_t i = 0; i < dim; ++i)
      for (const auto& child : node->children(i)) queue.emplace(child);
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

template <typename I, typename Ts, size_t dim>
template <typename FuncFilt, typename FuncPost>
inline void MultiNodeViewInterface<I, Ts, dim>::DeepRefine(
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

template <typename I, typename Ts, size_t dim>
inline void MultiNodeViewInterface<I, Ts, dim>::UniformRefine(
    std::array<int, dim> max_levels) {
  DeepRefine([&](const Ts& nodes) {
    bool result = true;
    static_for<dim>([&](auto i) {
      result = result && (std::get<i>(nodes)->level() <= max_levels[i]);
    });
    return result;
  });
}

template <typename I, typename Ts, size_t dim>
inline void MultiNodeViewInterface<I, Ts, dim>::SparseRefine(
    int max_level, std::array<int, dim> weights) {
  DeepRefine([&](const Ts& nodes) {
    auto levels = this->levels(nodes);
    int w_level = 0;
    for (int i = 0; i < dim; ++i) w_level += weights[i] * levels[i];
    return w_level <= max_level;
  });
}

template <typename I, typename Ts, size_t dim>
template <typename I_other, typename FuncFilt, typename FuncPost>
void MultiNodeViewInterface<I, Ts, dim>::Union(
    std::shared_ptr<I_other> other, const FuncFilt& call_filter,
    const FuncPost& call_postprocess) {
  assert(is_root() && other->is_root());
  assert(self().nodes() == other->nodes());
  std::queue<std::pair<std::shared_ptr<I>, std::shared_ptr<I_other>>> queue;
  queue.emplace(self().shared_from_this(), other);
  std::vector<std::shared_ptr<I>> my_nodes;
  while (!queue.empty()) {
    std::shared_ptr<I> my_node;
    std::shared_ptr<I_other> other_node;
    std::tie(my_node, other_node) = queue.front();
    queue.pop();
    assert(my_node->nodes() == other_node->nodes());
    if (my_node->marked()) continue;

    call_postprocess(my_node, other_node);
    my_node->set_marked(true);
    my_nodes.emplace_back(my_node);

    // Now do the union magic in all dimensions.
    static_for<dim>([&queue, &my_node, &other_node, &call_filter](auto i) {
      // Get a list of all children of the other_node in axis `i`.
      std::vector<std::tuple_element_t<i, Ts>> other_children_i;
      for (const auto& other_child_i : other_node->children(i))
        other_children_i.emplace_back(std::get<i>(other_child_i->nodes()));

      // Refine my_node using this list of children.
      my_node->template Refine<i>(other_children_i, call_filter,
                                  /*make_conforming*/ false);

      // Now only put children on the queue that other_node has as well.
      for (const auto& my_child : my_node->children(i))
        for (const auto& other_child : other_node->children(i))
          if (my_child->nodes() == other_child->nodes()) {
            queue.emplace(my_child, other_child);
            break;
          }
    });
  }
  // Reset mark field.
  for (const auto& my_node : my_nodes) {
    my_node->set_marked(false);
  }
}

template <typename I, typename Ts, size_t dim>
template <typename I_other, typename FuncPost>
inline std::shared_ptr<I_other> MultiNodeViewInterface<I, Ts, dim>::DeepCopy(
    const FuncPost& call_postprocess) {
  assert(self().is_root());
  auto new_root = std::apply(I_other::CreateRoot, self().nodes());
  new_root->Union(self().shared_from_this(), /*call_filter*/ func_true,
                  call_postprocess);
  return new_root;
}

template <typename I, typename Ts, size_t dim>
template <size_t i, typename Func>
inline const std::vector<std::shared_ptr<I>>&
MultiNodeViewInterface<I, Ts, dim>::Refine(
    const std::vector<std::tuple_element_t<i, Ts>>& children_i,
    const Func& call_filter, bool make_conforming) {
  static_assert(i < dim);
  if (is_full<i>()) return self().children(i);

  const auto& nodes = self().nodes();
  const auto& nodes_i = std::get<i>(nodes);
  for (const auto& child_i : children_i) {
    // Assert that this child lies in the underlying tree.
    assert(std::find(nodes_i->children().begin(), nodes_i->children().end(),
                     child_i) != nodes_i->children().end());

    // Create a copy of nodes, with the i-th index replaced by child_i.
    auto child_nodes{nodes};
    std::get<i>(child_nodes) = child_i;

    // Skip if this child already exists.
    if (std::any_of(self().children(i).begin(), self().children(i).end(),
                    [child_nodes](const auto& child) {
                      return child->nodes() == child_nodes;
                    }))
      continue;

    // Also skip if the call_filter doesn't pass.
    if (!call_filter(child_nodes)) continue;

    // Collect all brothers.
    std::array<std::vector<std::shared_ptr<I>>, dim> brothers;

    // Find brothers in all axes, using a static for loop over j.
    static_for<dim>([make_conforming, &brothers, &child_nodes, this](auto j) {
      for (const auto& child_parent_j : std::get<j>(child_nodes)->parents()) {
        // Create a copy of the child_nodes, replacing j-th index.
        auto brother_nodes{child_nodes};
        std::get<j>(brother_nodes) = child_parent_j;
        brothers[j].push_back(
            FindBrother<i, j>(brother_nodes, make_conforming));
      }
    });

    // Now finally, lets create an actual child!
#ifndef BOOST_ALLOCATOR
    auto child = std::make_shared<I>(/*nodes*/ std::move(child_nodes),
                                     /*parents*/ brothers);
#else
    typedef boost::fast_pool_allocator<I> BoostAlloc;
    auto child = std::allocate_shared<I, BoostAlloc>(
        BoostAlloc(), /*nodes*/ std::move(child_nodes),
        /*parents*/ brothers);
#endif

    // Add this child to all brothers.
    for (size_t j = 0; j < dim; ++j) {
      for (const auto& brother : brothers[j]) {
        brother->children(j).push_back(child);
      }
    }
  }
  return self().children(i);
}

template <typename I, typename Ts, size_t dim>
template <size_t i, size_t j>
inline std::shared_ptr<I> MultiNodeViewInterface<I, Ts, dim>::FindBrother(
    const Ts& nodes, bool make_conforming) {
  if (nodes == self().nodes()) {
    return self().shared_from_this();
  }

  for (const auto& parent_j : self().parents(j)) {
    for (const auto& sibling_i : parent_j->children(i)) {
      if (sibling_i->nodes() == nodes) {
        return sibling_i;
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
      parent_j->template Refine<i>(std::vector{nodes_i},
                                   /*call_filter*/ func_true,
                                   /*make_conforming*/ true);
    }
  }

  // Try calling this function again.
  return FindBrother<i, j>(nodes, /*make_conforming*/ false);
}

};  // namespace datastructures
