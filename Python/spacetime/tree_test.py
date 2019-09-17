import random
from collections import deque
from pprint import pprint

import pytest

from tree import *


class FakeNode(Node):
    """ Fake node. Implements some basic funcionality. """
    def __init__(self, labda, node_type, parents=None, children=None):
        super().__init__(labda=labda, parents=parents, children=children)
        self.node_type = node_type

    @property
    def level(self):
        return self.labda[0]

    def __repr__(self):
        return "({}, {}, {})".format(self.node_type, *self.labda)


class FakeHaarNode(FakeNode):
    """ Fake haar node. Proper tree structure with two children. """
    @property
    def support(self):
        l, n = self.labda
        return (n * 2**-l, (n + 1) * 2**-l)

    def refine(self):
        if self.children: return
        l, n = self.labda
        self.children.append(
            FakeHaarNode((l + 1, 2 * n), self.node_type, [self]))
        self.children.append(
            FakeHaarNode((l + 1, 2 * n + 1), self.node_type, [self]))
        return self.children

    def is_full(self):
        return len(self.children) in [0, 2]


class FakeOrthoNode(FakeNode):
    """ Fake orthonormal node. Familytree structure with 4 children, 2 parents. """
    def __init__(self, labda, node_type, parents=None, children=None):
        super().__init__(labda, node_type, parents, children)
        self.node_type = node_type
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
        left_0 = FakeOrthoNode((l + 1, 2 * n), self.node_type, parents)
        left_1 = FakeOrthoNode((l + 1, 2 * n + 1), self.node_type, parents)
        right_0 = FakeOrthoNode((l + 1, 2 * n + 2), self.node_type, parents)
        right_1 = FakeOrthoNode((l + 1, 2 * n + 3), self.node_type, parents)
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


class DebugDoubleNode(DoubleNode):
    total_counter = 0
    idx_counter = defaultdict(int)

    def __init__(self, nodes, parents=None, children=None):
        super().__init__(nodes, parents, children)

        if isinstance(nodes[0], MetaRoot) or isinstance(nodes[1], MetaRoot):
            return
        # For debugging purposes
        DebugDoubleNode.total_counter += 1
        DebugDoubleNode.idx_counter[(nodes[0].labda, nodes[1].labda)] += 1

    @property
    def level(self):
        l = -1 if isinstance(self.nodes[0], MetaRoot) else self.nodes[0].level
        l += -1 if isinstance(self.nodes[1], MetaRoot) else self.nodes[1].level
        return l


def create_roots(node_type, node_class):
    """ Returns a MetaRoot instance, containing all necessary roots. """
    if node_class is FakeHaarNode:
        return MetaRoot(FakeHaarNode((0, 0), node_type))
    elif node_class is FakeOrthoNode:
        root_0 = FakeOrthoNode((0, 0), node_type)
        root_1 = FakeOrthoNode((0, 1), node_type)
        root_0.nbr = root_1
        root_1.nbr = root_0
        return MetaRoot([root_0, root_1])
    else:
        assert False


def uniform_index_tree(max_level, node_type, node_class):
    """ Creates a (dummy) index tree with 2**l elements per level.
    
    Creates a field node_type inside the nodes and sets it to the node_type.
    """
    tree = create_roots(node_type, node_class)
    Lambda_l = tree.roots.copy()
    for _ in range(max_level):
        Lambda_new = []
        for node in Lambda_l:
            Lambda_new.extend(node.refine())
        Lambda_l = Lambda_new
    return tree


def corner_index_tree(max_level,
                      node_type,
                      which_child=0,
                      node_class=FakeHaarNode):
    """ Creates a (dummy) index tree with 1 element per level.
    
    Creates a field node_type inside the nodes and sets it to the node_type.
    """
    tree = create_roots(node_type, node_class)
    Lambda_l = tree.roots.copy()
    for _ in range(max_level):
        Lambda_new = []
        for node in Lambda_l:
            children = node.refine()
            Lambda_new.append(children[which_child])
        Lambda_l = Lambda_new
    return tree


def full_tensor_double_tree(meta_root_time, meta_root_space, max_levels=None):
    """ Makes a full grid doubletree from the given single trees. """
    double_root = DebugDoubleNode((meta_root_time, meta_root_space))
    queue = deque()
    queue.append(double_root)
    while queue:
        double_node = queue.popleft()
        for i in [0, 1]:
            if max_levels and double_node.nodes[i].level >= max_levels[i]:
                continue
            if double_node.children[i]: continue
            children = double_node.refine(i)
            queue.extend(children)

    return DoubleTree(double_root)


def uniform_full_grid(time_level, space_level, node_class=FakeHaarNode):
    """ Makes a full grid doubletree of uniformly refined singletrees. """
    meta_root_time = uniform_index_tree(time_level, 't', node_class)
    meta_root_space = uniform_index_tree(space_level, 'x', node_class)
    return full_tensor_double_tree(meta_root_time, meta_root_space)


def sparse_tensor_double_tree(meta_root_time, meta_root_space, max_level):
    """ Makes a sparse grid doubletree from the given single trees. """
    double_root = DebugDoubleNode((meta_root_time, meta_root_space))
    queue = deque()
    queue.append(double_root)
    while queue:
        double_node = queue.popleft()

        if double_node.level >= max_level: continue
        for i in [0, 1]:
            if double_node.children[i]: continue
            queue.extend(double_node.refine(i))

    return DoubleTree(double_root)


def uniform_sparse_grid(max_level, node_class):
    """ Makes a sparse grid doubletree of uniformly refined singletrees. """
    meta_root_time = uniform_index_tree(max_level, 't', node_class)
    meta_root_space = uniform_index_tree(max_level, 'x', node_class)
    return sparse_tensor_double_tree(meta_root_time, meta_root_space,
                                     max_level)


def random_double_tree(meta_root_time, meta_root_space, max_level, N):
    """ Makes a random doubletree from the given single trees. """
    tree = sparse_tensor_double_tree(meta_root_time, meta_root_space,
                                     max_level)
    n = 0
    leaves = set([n for n in tree.bfs() if n.is_leaf()])
    while n < N and not tree.root.is_leaf():
        node = leaves.pop()
        parents = node.parents[0] + node.parents[1]
        node.coarsen()
        for parent in parents:
            if parent.is_leaf():
                leaves.add(parent)
        n += 1
    return DoubleTree(tree.root)


def test_uniform_index_tree():
    root_haar = uniform_index_tree(5, 't', FakeHaarNode)
    assert len(root_haar.bfs()) == 2**6
    root_ortho = uniform_index_tree(5, 't', FakeOrthoNode)
    assert len(root_ortho.bfs()) == 2**7 - 1


def test_full_tensor():
    DebugDoubleNode.total_counter = 0
    double_root = uniform_full_grid(4, 2, FakeHaarNode)
    assert DebugDoubleNode.total_counter == (2**5 - 1) * (2**3 - 1)
    DebugDoubleNode.total_counter = 0
    double_root = uniform_full_grid(4, 2, FakeOrthoNode)
    assert DebugDoubleNode.total_counter == (2**6 - 2) * (2**4 - 2)


def test_sparse_tensor():
    DebugDoubleNode.total_counter = 0
    double_root = uniform_sparse_grid(1, FakeHaarNode)
    assert DebugDoubleNode.total_counter == 5
    DebugDoubleNode.total_counter = 0
    double_root = uniform_sparse_grid(4, FakeHaarNode)
    assert DebugDoubleNode.total_counter == 129

    DebugDoubleNode.total_counter = 0
    double_root = uniform_sparse_grid(1, FakeOrthoNode)
    assert DebugDoubleNode.total_counter == 2 * 2 + 2 * 2 * 4


def test_meta_root_refine():
    meta_root_time = uniform_index_tree(2, 't', FakeHaarNode)
    meta_root_space = uniform_index_tree(2, 'x', FakeHaarNode)
    root = DebugDoubleNode((meta_root_time, meta_root_space))
    # Create the real root
    root.refine(0)
    root.refine(1)
    root.refine(0)[0].refine(0)
    root.refine(1)[0].refine(0)
    real_root = root.children[0][0].children[1][0]

    left, right = real_root.refine(0)
    with pytest.raises(AssertionError):
        left.refine(1)


def test_project():
    for node_cls in [FakeHaarNode, FakeOrthoNode]:
        meta_root_time = uniform_index_tree(2, 't', node_cls)
        meta_root_space = uniform_index_tree(3, 'x', node_cls)
        for tree in [
                full_tensor_double_tree(meta_root_time, meta_root_space),
                sparse_tensor_double_tree(meta_root_time, meta_root_space, 5)
        ]:
            assert tree.project(0).bfs() == meta_root_time.bfs()
            assert tree.project(1).bfs() == meta_root_space.bfs()


def test_fiber():
    def slow_fiber(i, mu):
        return [
            node.nodes[i] for node in tree.bfs() if node.nodes[not i] is mu
        ]

    for cls in [FakeHaarNode, FakeOrthoNode]:
        for tree in [
                full_tensor_double_tree(corner_index_tree(4, 't', 0, cls),
                                        corner_index_tree(4, 'x', 1, cls)),
                sparse_tensor_double_tree(corner_index_tree(4, 't', 0, cls),
                                          corner_index_tree(4, 'x', 1, cls),
                                          8),
                sparse_tensor_double_tree(corner_index_tree(4, 't', 0, cls),
                                          uniform_index_tree(4, 'x', cls), 8),
                random_double_tree(uniform_index_tree(4, 't', cls),
                                   uniform_index_tree(4, 'x', cls),
                                   7,
                                   N=500),
        ]:
            for i in [0, 1]:
                for mu in tree.root.bfs(not i):
                    assert tree.fiber(i, mu.nodes[not i]).bfs() == slow_fiber(
                        i, mu.nodes[not i])


def test_union():
    """ Test that union indeed copies a tree. """
    meta_root_time = corner_index_tree(7, 't', 0)
    meta_root_space = corner_index_tree(7, 'x', 1)
    from_tree = full_tensor_double_tree(meta_root_time, meta_root_space)
    to_tree = DoubleTree(DoubleNode((meta_root_time, meta_root_space)))

    assert len(to_tree.bfs()) == 1
    # Copy axis 0 into `to_tree`.
    to_tree.root.union(from_tree.project(0), 0)
    assert len(to_tree.bfs()) == len(meta_root_time.bfs())

    # Copy all subtrees in axis 1 into `to_tree`.
    for item in to_tree.root.bfs(0):
        item.union(from_tree.fiber(1, item.nodes[0]), 1)
    assert len(to_tree.bfs()) == len(from_tree.bfs())

    # Assert double-tree structure is copied as well.
    assert to_tree.root.children[0][0].children[1][0] == \
           to_tree.root.children[1][0].children[0][0]
