from collections import defaultdict
from itertools import product
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from ..datastructures.double_tree import DoubleTree
from ..datastructures.double_tree_test import (corner_index_tree,
                                               full_tensor_double_tree,
                                               sparse_tensor_double_tree,
                                               uniform_index_tree)
from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 FrozenDoubleNodeVector)
from ..datastructures.function_test import FakeHaarFunction
from ..space.triangulation import InitialTriangulation
from ..time.haar_basis import HaarBasis
from .applicator import Applicator


class FakeApplicator(Applicator):
    class FakeSingleApplicator:
        def __init__(self, axis):
            self.axis = axis

        def apply(vec, fiber_in, fiber_out):
            return {}

    def __init__(self, Lambda_in, Lambda_out=None):
        super().__init__(None, Lambda_in, self.FakeSingleApplicator('t'),
                         self.FakeSingleApplicator('x'), None, Lambda_out)


class FakeHaarFunctionExt(FakeHaarFunction):
    """ Extend the FakeHaarFunction by fields necessary for the applicator. """
    def __init__(self, labda, f_type, parents=None, children=None):
        super().__init__(labda, f_type, parents, children)
        self.Sigma_psi_out = []

    @property
    def support(self):
        return [self]


def test_small_sigma():
    """ I computed on a piece of paper what Sigma should be for this combo. """
    root_time = uniform_index_tree(1, 't', node_class=FakeHaarFunctionExt)
    root_space = uniform_index_tree(1, 'x', node_class=FakeHaarFunctionExt)
    Lambda_in = full_tensor_double_tree(root_time, root_space)
    Lambda_out = full_tensor_double_tree(root_time, root_space)
    applicator = FakeApplicator(Lambda_in, Lambda_out)
    sigma = applicator.sigma()
    assert [n.nodes[0].labda for n in sigma.bfs()] == [(0, 0), (0, 0), (0, 0)]
    assert [n.nodes[1].labda for n in sigma.bfs()] == [(0, 0), (1, 0), (1, 1)]


def test_applicator_real():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(2)

    # Create time part.
    HaarBasis.metaroot_wavelet.uniform_refine(2)

    # Create a DoubleTree Vector
    dt_root = DoubleTree.full_tensor(HaarBasis.metaroot_wavelet,
                                     triang.vertex_meta_root,
                                     dbl_node_cls=DoubleNodeVector,
                                     frozen_dbl_cls=FrozenDoubleNodeVector)

    # Initialize it to random values
    for db_node in dt_root.bfs():
        db_node.value = np.random.rand()
    pprint(dt_root.bfs())


#    DoubleTreePlotter.plot_matplotlib_graph(dt_root, i_in=0)
#    DoubleTreePlotter.plot_matplotlib_graph(dt_root, i_in=1)
#    plt.show()
#    root_time = uniform_index_tree(2, 't', node_class=HaarWavelet)
#    root_space = uniform_index_tree(2, 'x', node_class=Vertex)
#    Lambda_in = full_tensor_double_tree(root_time, root_space)
#    Lambda_out = full_tensor_double_tree(root_time, root_space)
#    applicator = FakeApplicator(Lambda_in, Lambda_out)
#    sigma = applicator.sigma()
#    assert [n.nodes[0].labda for n in sigma.bfs()] == [(0, 0), (0, 0), (0, 0)]
#    assert [n.nodes[1].labda for n in sigma.bfs()] == [(0, 0), (1, 0), (1, 1)]


def test_sigma_combinations():
    """ I have very little intuition for what Sigma does, exactly, so I just
    made a bunch of weird combinations of Lambda_in and Lambda_out, made Sigma,
    and hoped that everything is all-right. """
    for (L_in, L_out) in product(product(range(0, 5), range(0, 5)),
                                 product(range(0, 5), range(0, 5))):
        Lmax = max(L_in[0], L_out[0]), max(L_in[1], L_out[1])
        for roots in product([
                uniform_index_tree(
                    Lmax[0], 't', node_class=FakeHaarFunctionExt),
                corner_index_tree(Lmax[0], 't', node_class=FakeHaarFunctionExt)
        ], [
                uniform_index_tree(
                    Lmax[1], 'x', node_class=FakeHaarFunctionExt),
                corner_index_tree(Lmax[1], 'x', node_class=FakeHaarFunctionExt)
        ]):
            Lambdas_in = [
                full_tensor_double_tree(*roots, L_in),
                sparse_tensor_double_tree(*roots, L_in[0])
            ]
            Lambdas_out = [
                full_tensor_double_tree(*roots, L_out),
                sparse_tensor_double_tree(*roots, L_out[0])
            ]
            for (Lambda_in, Lambda_out) in product(Lambdas_in, Lambdas_out):
                applicator = FakeApplicator(Lambda_in, Lambda_out)
                sigma = applicator.sigma()
                for node in sigma.bfs():
                    assert all(
                        node.nodes[i].is_full() or node.nodes[i].is_leaf()
                        for i in [0, 1])
