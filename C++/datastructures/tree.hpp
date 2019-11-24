#pragma once
#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

#include "cassert"

namespace datastructures {
template <typename I>
class NodeInterface : public std::enable_shared_from_this<I> {
 public:
  template <typename Func>
  std::vector<std::shared_ptr<I>> Bfs(bool include_metaroot, Func &&callback) {
    std::vector<std::shared_ptr<I>> nodes;
    std::queue<std::shared_ptr<I>> queue;
    queue.emplace(static_cast<I *>(this)->shared_from_this());
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
    if (!include_metaroot) {
      nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                                 [](const std::shared_ptr<I> &node) {
                                   return node->is_metaroot();
                                 }),
                  nodes.end());
    }
    return nodes;
  }

  std::vector<std::shared_ptr<I>> Bfs(bool include_metaroot = false) {
    return Bfs(include_metaroot, [](std::shared_ptr<I> x) {});
  }

  void UniformRefine(int max_level) {
    Bfs(true, [max_level](std::shared_ptr<I> node) {
      if (node->level() < max_level) {
        node->refine();
      }
    });
  }
};

template <typename I>
class Node : public NodeInterface<I> {
 public:
  explicit Node(const std::vector<std::shared_ptr<I>> &parents)
      : parents_(parents) {
    children_.reserve(2);
    assert(parents.size());
    level_ = parents[0]->level() + 1;
    for (const auto &parent : parents) {
      assert(parent->level() == level_ - 1);
    }
  }

  int level() const { return level_; }
  bool marked() const { return marked_; }
  void set_marked(bool value) { marked_ = value; }
  bool is_metaroot() const { return (level_ == -1); }
  bool is_leaf() const { return children_.size() == 0; }
  const std::vector<std::shared_ptr<I>> &parents() const { return parents_; }
  std::vector<std::shared_ptr<I>> &children() { return children_; }

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

 protected:
  int level_;
  bool marked_ = false;
  void *data_ = nullptr;
  std::vector<std::shared_ptr<I>> parents_;
  std::vector<std::shared_ptr<I>> children_;
  Node() : level_(-1) {}
};

template <typename I>
class BinaryNode : public Node<I> {
 public:
  explicit BinaryNode(std::shared_ptr<I> parent) : Node<I>({parent}) {}

  bool is_full() const { return children_.size() == 2; }
  std::shared_ptr<I> parent() const { return parents_[0]; }

 protected:
  using Node<I>::Node;
  using Node<I>::parents_;
  using Node<I>::children_;
};
}  // namespace datastructures
