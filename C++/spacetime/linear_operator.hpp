#pragma once
#include "datastructures/double_tree_view.hpp"
#include "space/basis.hpp"
#include "time/bases.hpp"

namespace spacetime {

// Interpolates a function into Z_delta.
template <typename Func>
Eigen::VectorXd Interpolate(
    Func &&g,
    const datastructures::DoubleTreeVector<
        Time::HierarchicalWaveletFn, space::HierarchicalBasisFn> &dbltree) {
  Eigen::VectorXd result = Eigen::VectorXd::Zero(dbltree.container().size());
  size_t i = 0;
  for (auto &dblnode : dbltree.container()) {
    i++;
    if (dblnode.is_metaroot()) continue;
    StaticVector<std::pair<double, double>, 3> eval_time;
    StaticVector<std::pair<std::pair<double, double>, double>, 3> eval_space;

    // Create dual time basis.
    eval_time.emplace_back(dblnode.node_0()->vertex(), 1.0);
    if (dblnode.node_0()->level()) {
      eval_time.emplace_back(dblnode.node_0()->Interval().first, -0.5);
      eval_time.emplace_back(dblnode.node_0()->Interval().second, -0.5);
    }

    // Create dual space basis.
    auto space_vertex = dblnode.node_1()->vertex();
    eval_space.emplace_back(std::pair{space_vertex->x, space_vertex->y}, 1.0);
    if (dblnode.node_1()->level()) {
      eval_space.emplace_back(std::pair{space_vertex->godparents[0]->x,
                                        space_vertex->godparents[0]->y},
                              -0.5);
      eval_space.emplace_back(std::pair{space_vertex->godparents[1]->x,
                                        space_vertex->godparents[1]->y},
                              -0.5);
    }

    for (auto [t, c_time] : eval_time)
      for (auto [pt, c_space] : eval_space) {
        auto [x, y] = pt;
        result[i - 1] += c_time * c_space * g(t, x, y);
      }
  }
  return result;
}

// Evaluates the (time) trace operator.
template <typename BasisTime>
datastructures::TreeVector<space::HierarchicalBasisFn> Trace(
    double t, const datastructures::DoubleTreeVector<
                  BasisTime, space::HierarchicalBasisFn> &dbltree) {
  datastructures::TreeVector<space::HierarchicalBasisFn> time_slice(
      dbltree.root()->node_1());

  for (auto psi_time : dbltree.Project_0()->Bfs()) {
    double time_val = psi_time->node()->Eval(t);
    if (time_val != 0) {
      time_slice.root()->Union(
          psi_time->FrozenOtherAxis(),
          /* call_filter*/ datastructures::func_true, /* call_postprocess*/
          [time_val](const auto &my_node, const auto &other_node) {
            my_node->set_value(my_node->value() +
                               time_val * other_node->value());
          });
    }
  }
  return time_slice;
}

}  // namespace spacetime
