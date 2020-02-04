#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"
#include "../time/haar_basis.hpp"
#include "../time/orthonormal_basis.hpp"
#include "../time/three_point_basis.hpp"

namespace spacetime {
datastructures::DoubleTreeView<Time::OrthonormalWaveletFn,
                               space::HierarchicalBasisFn>
GenerateYDelta(const datastructures::DoubleTreeView<
               Time::ThreePointWaveletFn, space::HierarchicalBasisFn> &X_delta);

template <class DblTreeIn, class DblTreeOut>
auto GenerateSigma(DblTreeIn &Lambda_in, DblTreeOut &Lambda_out);

template <class DblTreeIn, class DblTreeOut>
auto GenerateTheta(DblTreeIn &Lambda_in, DblTreeOut &Lambda_out);
};  // namespace spacetime

#include "basis.ipp"
