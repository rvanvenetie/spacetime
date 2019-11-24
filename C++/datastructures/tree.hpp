#pragma once
#include <algorithm>
#include <deque>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "cassert"

namespace datastructures {

template <typename I>
class NodeInterface {
 public:
  template <typename Func>
  std::vector<I *> Bfs(bool include_metaroot, Func &&callback,
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

  std::vector<I *> Bfs(bool include_metaroot = false) {
    return Bfs(include_metaroot, [](I *x) {});
  }
};

template <typename I>
class Node : public NodeInterface<I> {
 public:
  explicit Node(const std::vector<I *> &parents) : parents_(parents) {
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
  const std::vector<I *> &parents() const { return parents_; }
  std::vector<I *> &children() { return children_; }

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
  std::vector<I *> parents_;
  std::vector<I *> children_;
  Node() : level_(-1) {}
};

template <typename I>
class BinaryNode : public Node<I> {
 public:
  explicit BinaryNode(I *parent) : Node<I>({parent}) {}

  bool is_full() const { return children_.size() == 2; }
  I *parent() const { return parents_[0]; }

 protected:
  using Node<I>::Node;
  using Node<I>::parents_;
  using Node<I>::children_;
};

template <typename I>
class Tree {
 public:
  I *meta_root;

  // This constructs the tree with a single meta_root.
  Tree() : nodes_() { meta_root = emplace_back(); }

  virtual const std::vector<I *> &Refine(I *node) = 0;
  void UniformRefine(int max_level) {
    meta_root->Bfs(true, [this, max_level](I *node) {
      if (node->level() < max_level) {
        Refine(node);
      }
    });
  }

  template <typename... Args>
  inline I *emplace_back(Args &&... args) {
    nodes_.push_back(I(std::forward<Args>(args)...));
    return &nodes_.back();
  }

 protected:
  std::deque<I> nodes_;
};

}  // namespace datastructures
