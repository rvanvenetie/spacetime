#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "multi_tree_view.hpp"

namespace datastructures {

// Helper functions for converting between (flattened) trees and vectors.
template <typename Iterable>
Eigen::VectorXd ToVector(const Iterable &nodes) {
  Eigen::VectorXd result(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    if constexpr (std::is_pointer_v<typename Iterable::value_type>)
      result[i] = nodes[i]->value();
    else
      result[i] = nodes[i].value();
  }
  return result;
}
template <typename Iterable>
void FromVector(const Iterable &nodes, const Eigen::VectorXd &vec) {
  assert(nodes.size() == vec.size());
  for (int i = 0; i < nodes.size(); ++i) {
    if constexpr (std::is_pointer_v<typename Iterable::value_type>)
      nodes[i]->set_value(vec[i]);
    else
      const_cast<typename Iterable::value_type &>(nodes[i]).set_value(vec[i]);
  }
}

template <typename I>
class MultiNodeVectorInterface {
 public:
  inline I *self() { return static_cast<I *>(this); }
  inline const I *self() const { return static_cast<const I *>(this); }

  void Reset() {
    for (const auto &node : self()->Bfs(true)) node->set_value(0);
  }

  // Note: this is not compatible with the Python ToArray!
  Eigen::VectorXd ToVector() const {
    return datastructures::ToVector(const_cast<I *>(self())->Bfs());
  }
  void FromVector(const Eigen::VectorXd &vec) {
    datastructures::FromVector(self()->Bfs(), vec);
  }
};

template <typename I, typename... T>
class MultiNodeVectorBase : public MultiNodeViewBase<I, T...>,
                            public MultiNodeVectorInterface<I> {
 public:
  using MultiNodeViewBase<I, T...>::MultiNodeViewBase;

  inline const double &value() const { return value_; }
  inline void set_value(double val) { value_ = val; }

 protected:
  double value_ = 0.0;
};

// Create instance of this class
template <typename... T>
class MultiNodeVector
    : public MultiNodeVectorBase<MultiNodeVector<T...>, T...> {
 public:
  using MultiNodeVectorBase<MultiNodeVector<T...>, T...>::MultiNodeVectorBase;
};

template <typename I>
class MultiTreeVector : public MultiTreeView<I> {
 private:
  using Super = MultiTreeView<I>;

 public:
  using MultiTreeView<I>::MultiTreeView;

  // Note: this is not compatible with the Python ToArray!
  Eigen::VectorXd ToVector() const { return Super::root_->ToVector(); }
  void FromVector(const Eigen::VectorXd &vec) { Super::root_->FromVector(vec); }

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
    for (auto &node : Super::multi_nodes_) node.set_value(0);
  }

  // Define some simple linear algebra functions.
  MultiTreeVector<I> &operator*=(double val) {
    for (auto &nv : Super::multi_nodes_) {
      nv.set_value(nv.value() * val);
    }
    return *this;
  }
  template <typename MT_other = MultiTreeVector<I>>
  MultiTreeVector<I> &operator+=(const MT_other &rhs) {
    Super::root_->Union(
        rhs.root(), /* call_filter*/ func_true, /* call_postprocess*/
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
using NodeVector = MultiNodeVector<T0>;

template <typename T0>
using TreeVector = MultiTreeVector<MultiNodeVector<T0>>;

template <typename T0, typename T1, typename T2>
using TripleTreeVector = MultiTreeVector<MultiNodeVector<T0, T1, T2>>;

}  // namespace datastructures
