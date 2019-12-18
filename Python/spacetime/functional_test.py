import itertools

import numpy as np
from pytest import approx

from ..datastructures.double_tree_vector import DoubleTreeVector
from ..datastructures.double_tree_view import DoubleTree
from ..space.basis import HierarchicalBasisFunction
from ..space.functional import Functional as FunctionalSpace
from ..space.operators import QuadratureOperator as QuadratureSpace
from ..space.triangulation import InitialTriangulation
from ..time.functional import Functional as FunctionalTime
from ..time.haar_basis import HaarBasis
from ..time.operators import quadrature as QuadratureTime
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis
from .functional import TensorFunctional


def test_tensor_functional_quadrature():
    # Create time bases.
    bases_time = [
        HaarBasis(),
        OrthonormalBasis(),
        ThreePointBasis(),
    ]
    for basis in bases_time:
        basis.metaroot_wavelet.uniform_refine(5)

    # Create space bases.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(5)
    hierarch_basis = HierarchicalBasisFunction.from_triangulation(triang)
    hierarch_basis.uniform_refine()

    g = [(
        lambda t: -2 * (1 + t**2),
        lambda xy: (xy[0] - 1) * xy[0] + (xy[1] - 1) * xy[1],
    ), (
        lambda t: 2 * t,
        lambda xy: (xy[0] - 1) * xy[0] * (xy[1] - 1) * xy[1],
    )]
    g_order = [(2, 2), (1, 4)]

    for basis_time, basis_space in itertools.product(bases_time,
                                                     [hierarch_basis]):
        print('\nTesting for basis_time={}, basis_space={}'.format(
            basis_time.__class__.__name__, basis_space.__class__.__name__))
        l_in = 3

        # Create Lambda_in for which we want to evaluate the functional.
        Lambda = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        Lambda.sparse_refine(l_in)
        print(
            '\tLambda is a sparse grid tree upto level {} with dofs {}'.format(
                l_in, len(Lambda.bfs())))

        # Compare various rhs'
        for ((g_time, g_space), (g_time_order,
                                 g_space_order)), deriv in itertools.product(
                                     zip(g, g_order), [False, True]):
            functional_time = FunctionalTime(
                QuadratureTime(g=g_time, g_order=g_time_order, deriv=deriv),
                basis=basis_time,
            )
            functional_space = FunctionalSpace(
                QuadratureSpace(g=g_space, g_order=g_space_order, deriv=deriv))

            functional = TensorFunctional(functional_time=functional_time,
                                          functional_space=functional_space)
            vec_eval = functional.eval(Lambda)

            # Compare this with doing old-fashioned quadrature.
            for db_node in vec_eval.bfs():
                time_quad = db_node.nodes[0].inner_quad(g=g_time,
                                                        g_order=g_time_order,
                                                        deriv=deriv)
                space_quad = db_node.nodes[1].inner_quad(g=g_space,
                                                         g_order=g_space_order,
                                                         deriv=deriv)
                true_ip = time_quad * space_quad
                assert true_ip == approx(db_node.value)
