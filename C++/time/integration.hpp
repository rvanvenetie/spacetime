#pragma once

#include "basis.hpp"

namespace Time {
double Integrate(const std::function<double(double)>& f, const Element1D& elem,
                 size_t degree);
}  // namespace Time
