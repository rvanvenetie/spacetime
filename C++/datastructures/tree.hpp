#pragma once
#include <algorithm>
#include <memory>
#include <queue>
#include <vector>
#include "cassert"

namespace datastructures {
template <typename T>
class NodeInterface : public std::enable_shared_from_this<T> {
  /**
virtual int level() const = 0;
virtual bool marked() = 0;
virtual void set_marked(bool value) = 0;
virtual bool is_full() const = 0;
virtual bool is_metaroot() const = 0;
virtual const std::vector<std::shared_ptr<T>> &children() const = 0;
virtual const std::vector<std::shared_ptr<T>> &parents() const = 0;
**/

 public:
  std::vector<std::shared_ptr<T>> Bfs(
      bool include_metaroot, std::function<void(std::shared_ptr<T>)> callback) {
    std::vector<std::shared_ptr<T>> nodes;
    std::queue<std::shared_ptr<T>> queue;
    queue.emplace(static_cast<T *>(this)->shared_from_this());
    while (!queue.empty()) {
      auto node = queue.front();
      queue.pop();
      if (node->marked()) continue;
      nodes.emplace_back(node);
      node->set_marked(true);
      callback(node);
      for (auto child : node->children()) queue.emplace(child);
    }
    for (auto node : nodes) {
      node->set_marked(false);
    }
    if (!include_metaroot) {
      nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                                 [](const std::shared_ptr<T> &node) {
                                   return node->is_metaroot();
                                 }),
                  nodes.end());
    }
    return nodes;
  }

  std::vector<std::shared_ptr<T>> Bfs(bool include_metaroot = false) {
    return Bfs(include_metaroot, [](std::shared_ptr<T> x) {});
  }

  void UniformRefine(int max_level) {
    Bfs(true, [max_level](std::shared_ptr<T> node) {
      if (node->level() < max_level) {
        node->refine();
      }
    });
  }
};

template <typename T>
class Node : public NodeInterface<T> {
 protected:
  int level_;
  bool marked_ = false;
  std::vector<std::shared_ptr<T>> parents_;
  std::vector<std::shared_ptr<T>> children_;
  Node() : level_(-1) {}

 public:
  int level() const { return level_; }
  bool marked() const { return marked_; }
  void set_marked(bool value) { marked_ = value; }
  bool is_metaroot() const { return (level_ == -1); }
  const std::vector<std::shared_ptr<T>> &parents() const { return parents_; }
  std::vector<std::shared_ptr<T>> &children() { return children_; }

  explicit Node(const std::vector<std::shared_ptr<T>> &parents)
      : parents_(parents) {
    assert(parents.size());
    level_ = parents[0]->level() + 1;
    for (auto parent : parents) {
      assert(parent->level() == level_ - 1);
    }
  }
};

template <typename T>
class BinaryNode : public Node<T> {
 protected:
  using Node<T>::Node;

 public:
  bool is_full() const { return Node<T>::children_.size() == 2; }
  bool is_leaf() const { return Node<T>::children_.size() == 0; }
  std::shared_ptr<T> parent() const { return Node<T>::parents_[0]; }

  explicit BinaryNode(std::shared_ptr<T> parent) : Node<T>({parent}) {}
};
}  // namespace datastructures
