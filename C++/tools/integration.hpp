#pragma once

#include <vector>

#include "../space/triangulation.hpp"
#include "../time/basis.hpp"

namespace tools {

double Integrate1D(const std::function<double(double)>& f,
                   const Time::Element1D& elem, size_t degree);

double Integrate2D(const std::function<double(double, double)>& f,
                   const space::Element2D& elem, size_t degree);
}  // namespace tools
