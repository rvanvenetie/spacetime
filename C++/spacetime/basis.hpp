#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"
#include "../time/orthonormal_basis.hpp"
#include "../time/three_point_basis.hpp"

namespace spacetime {
datastructures::DoubleTreeView<Time::OrthonormalWaveletFn,
                               space::HierarchicalBasisFn>
GenerateYDelta(const datastructures::DoubleTreeView<
               Time::ThreePointWaveletFn, space::HierarchicalBasisFn> &X_delta);
};  // namespace spacetime
