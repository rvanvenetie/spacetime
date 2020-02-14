#pragma once

#include <stddef.h>

namespace space {
template <size_t order, class F, typename I>
void ApplyQuadrature(const F &f, I *root, bool dirichlet_boundary = true);
}  // namespace space

#include "linear_form.ipp"
