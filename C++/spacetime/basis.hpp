#pragma once

#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"
#include "../time/orthonormal_basis.hpp"
#include "../time/three_point_basis.hpp"

namespace spacetime {
template <typename DblTreeIn>
datastructures::DoubleTreeView<Time::OrthonormalWaveletFn,
                               space::HierarchicalBasisFn>
GenerateYDelta(const DblTreeIn &X_delta);

template <typename DblTreeIn, typename DblTreeOut = DblTreeIn>
DblTreeOut GenerateXDeltaUnderscore(const DblTreeIn &X_delta,
                                    size_t num_repeats = 1);

template <class DblTreeIn, class DblTreeOut>
auto GenerateSigma(const DblTreeIn &Lambda_in, const DblTreeOut &Lambda_out);

template <class DblTreeIn, class DblTreeOut>
auto GenerateTheta(const DblTreeIn &Lambda_in, const DblTreeOut &Lambda_out);
};  // namespace spacetime

#include "basis.ipp"
