from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt

from ..datastructures.double_tree import DoubleTree
from ..datastructures.double_tree_test import (corner_index_tree,
                                               full_tensor_double_tree,
                                               sparse_tensor_double_tree,
                                               uniform_index_tree)
from ..datastructures.function_test import FakeHaarFunction
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
