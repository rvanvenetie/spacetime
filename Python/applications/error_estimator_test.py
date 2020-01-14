import numpy as np

from ..datastructures.double_tree_vector import DoubleTreeVector
from ..space.basis import HierarchicalBasisFunction
from ..space.functional import Functional
from ..space.operators import QuadratureFunctional
from ..space.triangulation import InitialTriangulation
from ..time.three_point_basis import ThreePointBasis
from .error_estimator import ResidualErrorEstimator


def test_mean_zero_transformation():
    # Create space part.
    triang = InitialTriangulation.unit_square(initial_refinement=1)
    triang.elem_meta_root.uniform_refine(8)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(4)

    # Create quadrature functional
    functional_space = Functional(
        QuadratureFunctional(lambda xy: 1,
                             g_order=1,
                             deriv=False,
                             dirichlet_boundary=True))

    for level in range(1, 4):
        vec = DoubleTreeVector.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        vec.sparse_refine(2 * level, weights=[2, 1])
        for psi_t in vec.project(0).bfs():
            vec_quad = functional_space.eval(psi_t.frozen_other_axis())
            for psi_s, psi_quad in zip(psi_t.frozen_other_axis().bfs(),
                                       vec_quad.bfs()):
                psi_s.value = psi_quad.value

        ResidualErrorEstimator.mean_zero_basis_transformation(vec)
        for nv in vec.bfs():
            if nv.nodes[1].level <= 0: continue
            if any(parent.nodes[1].on_domain_boundary
                   for parent in nv.parents[1]):
                continue
            assert np.allclose(nv.value, 0)
