#pragma once

#include "../space/triangulation.hpp"

namespace space {

double Integrate(const std::function<double(double, double)>& f,
                 const Element2D& elem, size_t degree);
}  // namespace space
