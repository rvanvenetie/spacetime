#pragma once
#include "includes.hpp"

namespace datastructures {
// Below are two convenient lambda functions.
constexpr auto func_noop = [](const auto &... x) {};
constexpr auto func_true = [](const auto &... x) { return true; };
constexpr auto func_false = [](const auto &... x) { return false; };
using T_func_noop = decltype(func_noop);
using T_func_true = decltype(func_true);
using T_func_false = decltype(func_false);

template <typename I>
class NodeInterface {
 public:
  template <typename Func = T_func_noop>
  std::vector<I *> Bfs(bool include_metaroot = false,
                       const Func &callback = func_noop,
                       bool return_nodes = true) {
    std::vector<I *> nodes;
    std::queue<I *> queue;
    queue.emplace(static_cast<I *>(this));
    while (!queue.empty()) {
      auto node = queue.front();
      queue.pop();
      if (node->marked()) continue;
      nodes.emplace_back(node);
      node->set_marked(true);
      callback(node);
      for (const auto &child : node->children()) queue.emplace(child);
    }
    for (const auto &node : nodes) {
      node->set_marked(false);
    }
    if (!return_nodes) return {};
    if (!include_metaroot) {
      nodes.erase(
          std::remove_if(nodes.begin(), nodes.end(),
                         [](const I *node) { return node->is_metaroot(); }),
          nodes.end());
    }
    return nodes;
  }
};

template <typename I>
class Node : public NodeInterface<I> {
 public:
  // The implementation must publish constexpr N_parents and N_children. This
  // will give us possible optimalisations :-).
  explicit Node(const std::vector<I *> &parents) : parents_(parents) {
    children_.reserve(I::N_children);
    assert(parents.size());
    level_ = parents[0]->level() + 1;
    container_ = parents[0]->container_;
    assert(container_);
    for (const auto &parent : parents) {
      assert(parent->level() == level_ - 1);
      assert(parent->container_ == container_);
    }
  }

  int level() const { return level_; }
  bool marked() const { return marked_; }
  void set_marked(bool value) { marked_ = value; }
  bool is_leaf() const { return children_.size() == 0; }
  inline bool is_metaroot() const { return (level_ == -1); }
  const std::vector<I *> &parents() const { return parents_; }
  const std::vector<I *> &children() const { return children_; }

  // General data field for universal storage.
  template <typename T>
  T *data() {
    assert(data_ != nullptr);
    return static_cast<T *>(data_);
  }

  template <typename T>
  void set_data(T *value) {
    assert(data_ == nullptr);
    data_ = static_cast<void *>(value);
  }
  void reset_data() {
    assert(data_ != nullptr);
    data_ = nullptr;
  }
  bool has_data() { return data_ != nullptr; }

 protected:
  int level_;
  bool marked_ = false;
  void *data_ = nullptr;

  // Store parents/children as raw pointers.
  std::vector<I *> parents_;
  std::vector<I *> children_;

  // Pointer to the deque that holds all the childen.
  std::deque<I> *container_ = nullptr;

  template <typename... Args>
  inline I *make_child(Args &&... args) {
    container_->emplace_back(std::forward<Args>(args)...);
    return &container_->back();
  }

  Node() : level_(-1) {}
};

template <typename I>
class BinaryNode : public Node<I> {
 public:
  static constexpr size_t N_parents = 1;
  static constexpr size_t N_children = 2;
  explicit BinaryNode(I *parent) : Node<I>({parent}) {}

  I *parent() const { return parents_[0]; }
  inline bool is_full() const {
    if (Node<I>::is_metaroot()) return true;
    return children_.size() == 2;
  }

 protected:
  using Node<I>::Node;
  using Node<I>::parents_;
  using Node<I>::children_;
};

template <typename I>
class Tree {
 public:
  // This constructs the tree with a single meta_root.
  Tree(I &&meta_root = I()) {
    nodes_.emplace_back(std::move(meta_root));
    meta_root_ = &nodes_.back();
    meta_root_->container_ = &nodes_;
  }
  template <typename... Args>
  Tree(Args &&... args) : Tree(I(std::forward<Args>(args)...)) {}

  Tree(const Tree &) = delete;
  Tree<I> &operator=(Tree<I> &&) = default;

  template <typename Func = T_func_noop>
  std::vector<I *> Bfs(bool include_metaroot = false,
                       const Func &callback = func_noop,
                       bool return_nodes = true) {
    return meta_root_->Bfs(include_metaroot, callback, return_nodes);
  }

  template <typename Func>
  void DeepRefine(const Func &refine_filter) {
    meta_root_->Bfs(true, [refine_filter](I *node) {
      if (refine_filter(node)) node->Refine();
    });
  }
  void UniformRefine(int max_level) {
    DeepRefine([max_level](auto node) { return node->level() < max_level; });
  }

  // This returns nodes in this tree, sliced by levels.
  std::vector<std::vector<I *>> NodesPerLevel() {
    std::vector<std::vector<I *>> result;
    for (const auto &node : meta_root_->Bfs()) {
      assert(node->level() >= 0 && node->level() <= result.size());
      if (node->level() == result.size()) {
        result.emplace_back();
      }
      result[node->level()].push_back(node);
    }
    return result;
  }

  I *meta_root() { return meta_root_; }

 protected:
  I *meta_root_ = nullptr;
  std::deque<I> nodes_;
};

}  // namespace datastructures
