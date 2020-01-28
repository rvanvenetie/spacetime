#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "multi_tree_view.hpp"

namespace datastructures {

class VectorElement {
 public:
  inline double value() const { return value_; }
  inline void set_value(double val) { value_ = val; }

 protected:
  double value_ = 0.0;
};

template <typename... T>
class MultiNodeVector : public MultiNodeViewBase<MultiNodeVector<T...>, T...>,
                        public VectorElement {
 public:
  using MultiNodeViewBase<MultiNodeVector<T...>, T...>::MultiNodeViewBase;
};

template <typename T>
class NodeVector : public NodeViewBase<NodeVector<T>, T>, public VectorElement {
 public:
  using NodeViewBase<NodeVector<T>, T>::NodeViewBase;

  friend std::ostream &operator<<(std::ostream &os, const NodeVector<T> &nv) {
    os << "(" << *nv.node() << ", " << nv.value() << ")";
    return os;
  }
};

template <typename I>
class MultiTreeVector : public MultiTreeView<I> {
 private:
  using Super = MultiTreeView<I>;

 public:
  using MultiTreeView<I>::MultiTreeView;

  // Note: this is not compatible with the Python ToArray!
  Eigen::VectorXd ToVector(bool include_metaroot = false) const {
    auto nodes = Super::Bfs(include_metaroot);
    Eigen::VectorXd result(nodes.size());
    for (const auto &node : nodes) result << node->value();
    return result;
  }
  void FromVector(const Eigen::VectorXd &vec, bool include_metaroot = false) {
    auto nodes = Super::Bfs(include_metaroot);
    assert(nodes.size() == vec.size());
    for (int i = 0; i < nodes.size(); ++i) nodes[i]->set_value(vec[i]);
  }

  // Create a deepcopy that copies the vector data as well.
  template <typename MT_other = MultiTreeVector<I>>
  MT_other DeepCopy() const {
    return Super::template DeepCopy<MT_other>(
        [&](const auto &new_node, const auto &old_node) {
          new_node->set_value(old_node->value());
        });
  }

  // Set all values to zero.
  void Reset() {
    for (const auto &node : Super::Bfs(true)) node->set_value(0);
  }

  // Define some simple linear algebra functions.
  MultiTreeVector<I> &operator*=(double val) {
    for (const auto &nv : Super::Bfs()) {
      nv->set_value(nv->value() * val);
    }
    return *this;
  }
  template <typename MT_other = MultiTreeVector<I>>
  MultiTreeVector<I> &operator+=(const MT_other &rhs) {
    Super::root->Union(
        rhs.root, /* call_filter*/ func_true, /* call_postprocess*/
        [](const auto &my_node, const auto &other_node) {
          my_node->set_value(my_node->value() + other_node->value());
        });
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const MultiTreeVector<I> &tree) {
    os << "{";
    for (auto mnv : tree.Bfs()) os << *mnv << " ";
    os << "}";
    return os;
  }
};

template <typename T0>
using TreeVector = MultiTreeVector<NodeVector<T0>>;

template <typename T0, typename T1, typename T2>
using TripleTreeVector = MultiTreeVector<MultiNodeVector<T0, T1, T2>>;

}  // namespace datastructures
