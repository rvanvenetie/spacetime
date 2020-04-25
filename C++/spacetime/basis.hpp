#pragma once
#include "includes.hpp"

namespace spacetime {
template <template <typename, typename> class DblTreeIn,
          template <typename, typename> class DblTreeOut = DblTreeIn>
DblTreeOut<Time::OrthonormalWaveletFn, space::HierarchicalBasisFn>
GenerateYDelta(const DblTreeIn<Time::ThreePointWaveletFn,
                               space::HierarchicalBasisFn> &X_delta);

datastructures::DoubleTreeVector<Time::ThreePointWaveletFn,
                                 space::HierarchicalBasisFn>
GenerateXDeltaUnderscore(
    const datastructures::DoubleTreeVector<Time::ThreePointWaveletFn,
                                           space::HierarchicalBasisFn> &X_delta,
    size_t num_repeats = 1);

template <class DblTreeIn, class DblTreeOut>
auto GenerateSigma(const DblTreeIn &Lambda_in, const DblTreeOut &Lambda_out);

template <class DblTreeIn, class DblTreeOut>
auto GenerateTheta(const DblTreeIn &Lambda_in, const DblTreeOut &Lambda_out);

// Template specializations for GenerateYDelta.
extern template datastructures::DoubleTreeView<Time::OrthonormalWaveletFn,
                                               space::HierarchicalBasisFn>
GenerateYDelta<datastructures::DoubleTreeView, datastructures::DoubleTreeView>(
    const datastructures::DoubleTreeView<Time::ThreePointWaveletFn,
                                         space::HierarchicalBasisFn> &X_delta);
extern template datastructures::DoubleTreeVector<Time::OrthonormalWaveletFn,
                                                 space::HierarchicalBasisFn>
GenerateYDelta<datastructures::DoubleTreeVector,
               datastructures::DoubleTreeVector>(
    const datastructures::DoubleTreeVector<
        Time::ThreePointWaveletFn, space::HierarchicalBasisFn> &X_delta);

};  // namespace spacetime

#include "basis.ipp"
