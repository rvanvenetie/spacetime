import random

import numpy as np

from ..datastructures.double_tree_view import DoubleTree
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.three_point_basis import ThreePointBasis
from .adaptive_heat_equation import AdaptiveHeatEquation
from .heat_equation import HeatEquation
from .heat_equation_test import example_rhs_functional


def test_dorfler_marking():
    class FakeNode:
        def __init__(self, value):
            self.value = value

        def __lt__(self, other):
            return self.value < other.value

    n = 100
    nodes = [FakeNode(random.random()) for _ in range(n)]
    l2_norm = np.sqrt(sum(fn.value**2 for fn in nodes))

    # Test theta == 0.
    assert len(AdaptiveHeatEquation.dorfler_marking(nodes, 0)) == 0

    # Test theta == 1
    assert AdaptiveHeatEquation.dorfler_marking(nodes,
                                                1) == sorted(nodes)[::-1]

    for theta in [0.3, 0.5, 0.7]:
        bulk_nodes = AdaptiveHeatEquation.dorfler_marking(nodes, theta)
        assert len(bulk_nodes) < len(nodes)

        bulk_l2_norm = np.sqrt(sum(fn.value**2 for fn in bulk_nodes))
        assert bulk_l2_norm >= theta * l2_norm

        # Check that shuffled gives same results
        random.shuffle(nodes)
        assert AdaptiveHeatEquation.dorfler_marking(nodes, theta) == bulk_nodes


def test_heat_error_reduction(theta=0.7):
    # Printing options.
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=10000)

    # Create space part.
    triang = InitialTriangulation.unit_square(initial_refinement=1)
    triang.elem_meta_root.uniform_refine(1)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()

    # Create X^\delta as a sparse grid.
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.uniform_refine(0)

    # Create rhs functionals
    g_functional, u0_functional = example_rhs_functional(HeatEquation(X_delta))

    # Create adaptive heat equation object.
    adaptive_heat_eq = AdaptiveHeatEquation(X_init=X_delta,
                                            g_functional=g_functional,
                                            u0_functional=u0_functional,
                                            theta=theta)

    # Solve
    sol, errors = adaptive_heat_eq.solve(max_iters=2)
    print(errors)
