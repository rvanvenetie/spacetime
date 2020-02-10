#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "multi_tree_view.hpp"

namespace datastructures {

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
    auto nodes = const_cast<I *>(self())->Bfs();
    Eigen::VectorXd result(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) result[i] = nodes[i]->value();
    return result;
  }
  void FromVector(const Eigen::VectorXd &vec) {
    auto nodes = self()->Bfs();
    assert(nodes.size() == vec.size());
    for (int i = 0; i < nodes.size(); ++i) nodes[i]->set_value(vec[i]);
  }

  // In case dim == 1, we add functionality to read data from the
  // underlying tree.
  void ReadFromTree() {
    for (const auto &nv : self()->Bfs()) {
      auto node = nv->node();
      if (node->has_data())
        nv->set_value(*node->template data<double>());
      else
        nv->set_value(0);
    }
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
  void Reset() { Super::root_->Reset(); }

  // Define some simple linear algebra functions.
  MultiTreeVector<I> &operator*=(double val) {
    for (const auto &nv : Super::Bfs()) {
      nv->set_value(nv->value() * val);
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
