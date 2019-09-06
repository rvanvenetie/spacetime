from applicator import Applicator
from tree import DoubleTree
from tree_test import uniform_full_grid


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

    def __init__(self,
                 labda,
                 node_type,
                 parents=None,
                 children=None,
                 support=None):
        super().__init__(labda, parents, children)
        self.node_type = node_type
        self.support = [self] or support

    def refine(self):
        if self.children: return
        l, n = self.labda
        self.children.append(
            DummyFunctionNode(
                (l + 1, 2 * n),
                self.node_type,
                [self],
            ))
        self.children.append(
            DummyFunctionNode((l + 1, 2 * n + 1), self.node_type, [self]))
        return self.children

    @property
    def level(self):
        return self.labda[0]

    def __repr__(self):
        return "({}, {}, {})".format(self.node_type, *self.labda)


def test_sigma():
    from_tree = DoubleTree(uniform_full_grid(4, 2))
    applicator = DummyApplicator(from_tree)
    print(applicator.sigma())
