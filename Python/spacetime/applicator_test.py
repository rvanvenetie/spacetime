from applicator import Applicator
from tree import DoubleTree, Node
from tree_test import uniform_full_grid
from tree_plotter import TreePlotter
import matplotlib.pyplot as plt


class DummyApplicator(Applicator):
    class SingleApplicator:
        def __init__(self, axis):
            self.axis = axis

        def apply(vec, fiber_in, fiber_out):
            return {}

    def __init__(self, Lambda_in, Lambda_out=None):
        super().__init__(None, Lambda_in, self.SingleApplicator('t'),
                         self.SingleApplicator('x'), None, Lambda_out)


class DummyFunctionNode(Node):
    """ Dummy nodes that refines into two children, representing a function. """

    def __init__(self, labda, node_type, parents=None, children=None):
        super().__init__(labda, parents, children)
        self.node_type = node_type

        self.psi_out = [self]

    def refine(self):
        if self.children: return
        l, n = self.labda
        self.children.extend([
            DummyFunctionNode((l + 1, 2 * n + i), self.node_type, [self])
            for i in [0, 1]
        ])
        return self.children

    @property
    def level(self):
        return self.labda[0]

    @property
    def support(self):
        return [self]

    def __repr__(self):
        return "({}, {}, {})".format(self.node_type, *self.labda)


def test_sigma():
    Labda = DoubleTree(uniform_full_grid(4, 2, node_class=DummyFunctionNode))
    assert len(Labda.bfs()) == 4
    applicator = DummyApplicator(Labda)
    sigma = applicator.sigma()
    treeplotter = TreePlotter(Labda)
    treeplotter.plot_matplotlib_graph(i_in=0)
    treeplotter.plot_matplotlib_graph(i_in=1)
    treeplotter = TreePlotter(sigma)
    treeplotter.plot_matplotlib_graph(i_in=0)
    treeplotter.plot_matplotlib_graph(i_in=1)
    plt.show()
