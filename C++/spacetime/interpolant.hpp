#pragma once
#include "datastructures/double_tree_view.hpp"
#include "space/basis.hpp"
#include "time/bases.hpp"

namespace spacetime {

template <typename Func>
void Interpolate(
    Func &&g,
    datastructures::DoubleTreeVector<Time::HierarchicalWaveletFn,
                                     space::HierarchicalBasisFn> *dbltree) {
  for (auto &dblnode : dbltree->container()) {
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

    double val = 0;
    for (auto [t, c_time] : eval_time)
      for (auto [pt, c_space] : eval_space) {
        auto [x, y] = pt;
        val += c_time * c_space * g(t, x, y);
      }
    dblnode.set_value(val);
  }
}

}  // namespace spacetime
