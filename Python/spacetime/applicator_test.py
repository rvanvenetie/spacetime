from collections import defaultdict
from itertools import product

from applicator import Applicator
from tree import DoubleTree, Node
from tree_test import uniform_index_tree, full_tensor_double_tree, corner_refined_index_tree
from tree_plotter import TreePlotter
import matplotlib.pyplot as plt


class FakeApplicator(Applicator):
    class FakeSingleApplicator:
        def __init__(self, axis):
            self.axis = axis

        def apply(vec, fiber_in, fiber_out):
            return {}

    def __init__(self, Lambda_in, Lambda_out=None):
        super().__init__(None, Lambda_in, self.FakeSingleApplicator('t'),
                         self.FakeSingleApplicator('x'), None, Lambda_out)


class FakeFunctionNode(Node):
    """ Fake nodes refining into 2, representing a function AND an element. """
    def __init__(self, labda, node_type, parents=None, children=None):
        super().__init__(labda, parents, children)
        self.node_type = node_type
        self.psi_out = []

    def refine(self):
        if self.children: return
        l, n = self.labda
        self.children.append(
            FakeFunctionNode((l + 1, 2 * n), self.node_type, [self]))
        self.children.append(
            FakeFunctionNode((l + 1, 2 * n + 1), self.node_type, [self]))
        return self.children

    def is_full(self):
        return len(self.children) in [0, 2]

    @property
    def level(self):
        return self.labda[0]

    @property
    def support(self):
        return [self]

    def __repr__(self):
        return "({}, {}, {})".format(self.node_type, *self.labda)


def test_sigma():

    for L_in in product(range(0, 5), range(0, 5)):
        for L_out in product(range(0, 5), range(0, 5)):
            for root_tree_time in [
                    uniform_index_tree(max(L_in[0], L_out[0]),
                                       't',
                                       node_class=FakeFunctionNode),
                    corner_refined_index_tree(max(L_in[0], L_out[0]),
                                              't',
                                              0,
                                              node_class=FakeFunctionNode)
            ]:
                for root_tree_space in [
                        uniform_index_tree(max(L_in[1], L_out[1]),
                                           'x',
                                           node_class=FakeFunctionNode),
                        corner_refined_index_tree(max(L_in[1], L_out[1]),
                                                  'x',
                                                  0,
                                                  node_class=FakeFunctionNode)
                ]:
                    Lambda_in = DoubleTree(
                        full_tensor_double_tree(root_tree_time,
                                                root_tree_space,
                                                max_levels=L_in))
                    Lambda_out = DoubleTree(
                        full_tensor_double_tree(root_tree_time,
                                                root_tree_space,
                                                max_levels=L_out))

                    applicator = FakeApplicator(Lambda_in, Lambda_out)
                    sigma = applicator.sigma()

                    max0 = 0
                    max1 = 0
                    for elem in sigma.bfs():
                        max0 = max(max0, elem.nodes[0].level)
                        max1 = max(max1, elem.nodes[1].level)
                    if max0 != min(L_in[0], max(0, L_out[0] - 1)) or (
                            max1 != (L_out[1] if L_out[0] != 0 else 0)):
                        print(L_in, L_out, (max0, max1))
