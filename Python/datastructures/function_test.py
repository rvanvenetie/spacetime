from .function import FunctionNode
from .tree import MetaRoot


class FakeFunctionNode(FunctionNode):
    """ Fake node. Implements some basic funcionality. """
    def __init__(self, labda, f_type, parents=None, children=None):
        super().__init__(labda=labda, parents=parents, children=children)
        self.f_type = f_type

    def __repr__(self):
        return "({}, {}, {})".format(self.f_type, *self.labda)


class FakeHaarFunction(FakeFunctionNode):
    """ Fake haar node. Proper tree structure with two children. """
    @property
    def support(self):
        l, n = self.labda
        return (n * 2**-l, (n + 1) * 2**-l)

    def refine(self):
        if self.children: return
        l, n = self.labda
        self.children.append(
            self.__class__((l + 1, 2 * n), self.f_type, [self]))
        self.children.append(
            self.__class__((l + 1, 2 * n + 1), self.f_type, [self]))
        return self.children

    def is_full(self):
        return len(self.children) in [0, 2]


class FakeOrthoFunction(FakeFunctionNode):
    """ Fake orthonormal node. Familytree structure with 4 children, 2 parents. """
    def __init__(self, labda, f_type, parents=None, children=None):
        super().__init__(labda, f_type, parents, children)
        self.nbr = None

        l, n = self.labda
        if l > 0: assert self.parents

    @property
    def support(self):
        l, n = labda
        return (n // 2 * 2**-l, (n // 2 + 1) * 2**-l)

    def refine(self):
        if self.children: return self.children
        l, n = self.labda
        type_0 = n % 2 == 0  # Store some type.
        if not type_0: return self.nbr.refine()
        parents = [self, self.nbr]

        # Create four children
        left_0 = FakeOrthoFunction((l + 1, 2 * n), self.f_type, parents)
        left_1 = FakeOrthoFunction((l + 1, 2 * n + 1), self.f_type, parents)
        right_0 = FakeOrthoFunction((l + 1, 2 * n + 2), self.f_type, parents)
        right_1 = FakeOrthoFunction((l + 1, 2 * n + 3), self.f_type, parents)
        self.children = [left_0, left_1, right_0, right_1]
        self.nbr.children = self.children

        # Update neighbouring relations.
        left_0.nbr = left_1
        left_1.nbr = left_0
        right_0.nbr = right_1
        right_1.nbr = right_0
        return self.children

    def is_full(self):
        return len(self.children) in [0, 4]


def test_haar_function():
    root = FakeHaarFunction((0, 0), 'haar')
    root.refine()
    assert root.is_full()
    assert root.children[0].labda == (1, 0)
    assert root.children[1].labda == (1, 1)


def test_ortho_function():
    roots = [
        FakeOrthoFunction((0, 0), 'ortho'),
        FakeOrthoFunction((0, 1), 'ortho')
    ]
    roots[0].nbr = roots[1]
    roots[1].nbr = roots[0]

    roots[0].refine()
    assert roots[0].is_full()
    assert roots[1].is_full()
    assert roots[0].children == roots[1].children
    assert roots[0].children[0].labda == (1, 0)
    assert roots[1].children[3].labda == (1, 3)


def test_haar_refine():
    meta_root = MetaRoot(FakeHaarFunction((0, 0), 'haar'))
    meta_root.uniform_refine(4)
    assert len(meta_root.bfs()) == 2**5 - 1
    for node in meta_root.bfs():
        assert node.level <= 4


def test_ortho_refine():
    roots = [
        FakeOrthoFunction((0, 0), 'ortho'),
        FakeOrthoFunction((0, 1), 'ortho')
    ]
    roots[0].nbr = roots[1]
    roots[1].nbr = roots[0]

    meta_root = MetaRoot(roots)
    meta_root.uniform_refine(4)
    assert len(meta_root.bfs()) == (2**6 - 2)
    for node in meta_root.bfs():
        assert node.level <= 4
