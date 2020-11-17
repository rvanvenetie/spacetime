#pragma once
#include <stddef.h>

#include <utility>

#include "triangulation.hpp"

namespace space {

template <typename Func, typename I>
void Interpolate(Func &&f, I *root) {
  for (auto node : root->Bfs()) {
    // Create dual space basis.
    StaticVector<std::pair<std::pair<double, double>, double>, 3> eval_space;
    auto space_vertex = node->node()->vertex();
    eval_space.emplace_back(std::pair{space_vertex->x, space_vertex->y}, 1.0);
    if (node->node()->level()) {
      eval_space.emplace_back(std::pair{space_vertex->godparents[0]->x,
                                        space_vertex->godparents[0]->y},
                              -0.5);
      eval_space.emplace_back(std::pair{space_vertex->godparents[1]->x,
                                        space_vertex->godparents[1]->y},
                              -0.5);
    }

    double val = 0;
    for (auto [pt, coeff] : eval_space) {
      auto [x, y] = pt;
      val += coeff * f(x, y);
    }
    node->set_value(val);
  }
}

}  // namespace space
