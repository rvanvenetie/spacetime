from collections import defaultdict
from itertools import product
from pprint import pprint

import matplotlib.pyplot as plt

from ..datastructures.double_tree import DoubleTree
from ..datastructures.double_tree_test import (corner_index_tree,
                                               full_tensor_double_tree,
                                               sparse_tensor_double_tree,
                                               uniform_index_tree)
from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 FrozenDoubleNodeVector)
from ..datastructures.function_test import FakeHaarFunction
from ..datastructures.tree_view import MetaRootView
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.haar_basis import HaarBasis
from ..time.three_point_basis import ThreePointBasis
from .applicator import Applicator


class FakeApplicator(Applicator):
    class FakeSingleApplicator:
        def __init__(self, axis):
            self.axis = axis

        def apply(self, vec_in, vec_out):
            """ This simply sets all output values to 1. """
            for labda in vec_out.bfs():
                labda.dbl_node.value = 1

        def apply_low(self, vec_in, vec_out):
            """ This simply sets all output values to 1. """
            for labda in vec_out.bfs():
                labda.dbl_node.value = 1

        def apply_upp(self, vec_in, vec_out):
            """ This simply sets all output values to 1. """
            for labda in vec_out.bfs():
                labda.dbl_node.value = 1

    def __init__(self, Lambda_in, Lambda_out=None):
        super().__init__(None, Lambda_in, self.FakeSingleApplicator('t'),
                         self.FakeSingleApplicator('x'), None, Lambda_out)


class FakeHaarFunctionExt(FakeHaarFunction):
    """ Extend the FakeHaarFunction by fields necessary for the applicator. """
    def __init__(self, labda, f_type, parents=None, children=None):
        super().__init__(labda, f_type, parents, children)
        self.Sigma_psi_out = []
        self.Theta_psi_in = False

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


def test_theta_full_tensor():
    HaarBasis.metaroot_wavelet.uniform_refine(4)
    Lambda_in = DoubleTree.full_tensor(HaarBasis.metaroot_wavelet,
                                       HaarBasis.metaroot_wavelet)
    Lambda_out = DoubleTree.full_tensor(HaarBasis.metaroot_wavelet,
                                        HaarBasis.metaroot_wavelet)
    applicator = FakeApplicator(Lambda_in, Lambda_out)
    theta = applicator.theta()
    assert [d_node.nodes for d_node in theta.bfs()
            ] == [d_node.nodes for d_node in Lambda_in.bfs()]


def test_theta_small():
    HaarBasis.metaroot_wavelet.uniform_refine(3)
    Lambda_in = DoubleTree(
        (HaarBasis.metaroot_wavelet, HaarBasis.metaroot_wavelet))
    Lambda_out = DoubleTree(
        (HaarBasis.metaroot_wavelet, HaarBasis.metaroot_wavelet))

    # Define the maximum levels
    lvls = (3, 1)

    # Refine Lambda_in in time towards the origin
    Lambda_in.deep_refine(
        call_filter=lambda d_node: d_node[0].level <= lvls[0] and d_node[1].
        level <= lvls[1] and (d_node[0].level < 1 or d_node[0].labda[1] == 0))

    # Refine Lambda_out in time towards the end of the interval
    Lambda_out.deep_refine(call_filter=lambda d_node: d_node[0].level <= lvls[
        0] and d_node[1].level <= lvls[1] and (d_node[0].level < 1 or d_node[
            0].labda[1] == 2**(d_node[0].labda[0] - 1) - 1))

    assert len(Lambda_in.bfs()) == len(Lambda_out.bfs())
    applicator = FakeApplicator(Lambda_in, Lambda_out)
    theta = applicator.theta()

    assert [d.node for d in theta.project(1).bfs()
            ] == [d.node for d in Lambda_in.project(1).bfs()]

    assert set(
        (n.nodes[0].labda, n.nodes[1].labda) for n in theta.bfs()) == set([
            ((0, 0), (0, 0)),
            ((0, 0), (1, 0)),
            ((1, 0), (0, 0)),
            ((1, 0), (1, 0)),
        ])


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


def test_applicator_real():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(2)

    # Create a hierarchical basis
    hierarch_basis = MetaRootView(metaroot=triang.vertex_meta_root,
                                  node_view_cls=HierarchicalBasisFunction)
    hierarch_basis.deep_refine()

    # Create time part for Lambda_in.
    HaarBasis.metaroot_wavelet.uniform_refine(2)

    # Create time part for Lambda_out.
    ThreePointBasis.metaroot_wavelet.uniform_refine(3)

    # Create Lambda_in/out and initialize the applicator.
    Lambda_in = DoubleTree.full_tensor(HaarBasis.metaroot_wavelet,
                                       hierarch_basis)
    Lambda_out = DoubleTree.full_tensor(ThreePointBasis.metaroot_wavelet,
                                        hierarch_basis)
    applicator = FakeApplicator(Lambda_in, Lambda_out)

    # Now create an vec_in and vec_out.
    vec_in = Lambda_in.deep_copy(dbl_node_cls=DoubleNodeVector,
                                 frozen_dbl_cls=FrozenDoubleNodeVector)
    vec_out = Lambda_out.deep_copy(dbl_node_cls=DoubleNodeVector,
                                   frozen_dbl_cls=FrozenDoubleNodeVector)

    assert len(vec_in.bfs()) == len(Lambda_in.bfs())
    assert all(n1.nodes == n2.nodes
               for n1, n2 in zip(vec_in.bfs(), Lambda_in.bfs()))

    # Initialize the input vector with ones.
    for db_node in vec_in.bfs():
        assert db_node.value == 0
        db_node.value = 1

    applicator.apply(vec_in, vec_out)
    assert all(d_node.value == 1 for d_node in vec_out.bfs())
