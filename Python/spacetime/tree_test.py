import random
from collections import deque
from pprint import pprint

import pytest

from tree import *


class FakeNode(Node):
    """ Fake nodes that refines into two children. """
    def __init__(self, labda, node_type, parents=None, children=None):
        super().__init__(labda, parents, children)
        self.node_type = node_type

    @property
    def support(self):
        return (self.labda[1] * 2**(-self.labda[0]),
                (self.labda[1] + 1) * 2**(-self.labda[0]))

    def refine(self):
        if self.children: return
        l, n = self.labda
        self.children.append(FakeNode((l + 1, 2 * n), self.node_type, [self]))
        self.children.append(
            FakeNode((l + 1, 2 * n + 1), self.node_type, [self]))
        return self.children

    @property
    def level(self):
        return self.labda[0]

    def is_full(self):
        return len(self.children) in [0, 2]

    def __repr__(self):
        return "({}, {}, {})".format(self.node_type, *self.labda)


class DebugDoubleNode(DoubleNode):
    total_counter = 0
    idx_counter = defaultdict(int)

    def __init__(self, nodes, parents=None, children=None):
        super().__init__(nodes, parents, children)
        # For debugging purposes
        DebugDoubleNode.total_counter += 1
        DebugDoubleNode.idx_counter[(nodes[0].labda, nodes[1].labda)] += 1

    @property
    def level(self):
        return self.nodes[0].level + self.nodes[1].level


def uniform_index_tree(max_level, node_type, node_class=FakeNode):
    """ Creates a (dummy) index tree with 2**l elements per level.
    
    Creates a field node_type inside the nodes and sets it to the node_type.
    """
    root = node_class((0, 0), node_type)
    Lambda_l = [root]
    for _ in range(max_level):
        Lambda_new = []
        for node in Lambda_l:
            Lambda_new.extend(node.refine())
        Lambda_l = Lambda_new
    return root


def corner_index_tree(max_level, node_type, which_child=0,
                      node_class=FakeNode):
    """ Creates a (dummy) index tree with 1 element per level.
    
    Creates a field node_type inside the nodes and sets it to the node_type.
    """
    root = node_class((0, 0), node_type)
    Lambda_l = [root]
    for _ in range(max_level):
        Lambda_new = []
        for node in Lambda_l:
            children = node.refine()
            Lambda_new.append(children[which_child])
        Lambda_l = Lambda_new
    return root


def full_tensor_double_tree(root_tree_time, root_tree_space, max_levels=None):
    """ Makes a full grid doubletree from the given single trees. """
    double_root = DebugDoubleNode((root_tree_time, root_tree_space))
    queue = deque()
    queue.append(double_root)
    while queue:
        double_node = queue.popleft()
        for i in [0, 1]:
            if max_levels and double_node.nodes[i].level >= max_levels[i]:
                continue
            if double_node.children[i]: continue
            queue.extend(double_node.refine(i))

    return double_root


def uniform_full_grid(time_level, space_level):
    """ Makes a full grid doubletree of uniformly refined singletrees. """
    root_tree_time = uniform_index_tree(time_level, 't')
    root_tree_space = uniform_index_tree(space_level, 'x')
    return full_tensor_double_tree(root_tree_time, root_tree_space)


def sparse_tensor_double_tree(root_tree_time, root_tree_space, max_level):
    """ Makes a sparse grid doubletree from the given single trees. """
    double_root = DebugDoubleNode((root_tree_time, root_tree_space))
    queue = deque()
    queue.append(double_root)
    while queue:
        double_node = queue.popleft()

        if double_node.level >= max_level: continue
        for i in [0, 1]:
            if double_node.children[i]: continue
            queue.extend(double_node.refine(i))

    return double_root


def uniform_sparse_grid(max_level):
    """ Makes a sparse grid doubletree of uniformly refined singletrees. """
    root_tree_time = uniform_index_tree(max_level, 't')
    root_tree_space = uniform_index_tree(max_level, 'x')
    return sparse_tensor_double_tree(root_tree_time, root_tree_space,
                                     max_level)


def random_double_tree(root_tree_time, root_tree_space, max_level, N):
    """ Makes a random doubletree from the given single trees. """
    root = sparse_tensor_double_tree(root_tree_time, root_tree_space,
                                     max_level)
    tree = DoubleTree(root)
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
    return root


def test_full_tensor():
    DebugDoubleNode.total_counter = 0
    double_root = uniform_full_grid(4, 2)
    assert DebugDoubleNode.total_counter == (2**5 - 1) * (2**3 - 1)


def test_sparse_tensor():
    DebugDoubleNode.total_counter = 0
    double_root = uniform_sparse_grid(1)
    assert DebugDoubleNode.total_counter == 5
    DebugDoubleNode.total_counter = 0
    double_root = uniform_sparse_grid(4)
    assert DebugDoubleNode.total_counter == 129


def test_tree_refine():
    root_tree_time = uniform_index_tree(2, 't')
    root_tree_space = uniform_index_tree(2, 'x')
    root = DebugDoubleNode((root_tree_time, root_tree_space))
    left, right = root.refine(0)
    with pytest.raises(AssertionError):
        left.refine(1)


def test_project():
    root_tree_time = uniform_index_tree(2, 't')
    root_tree_space = uniform_index_tree(3, 'x')
    for db_root in [
            full_tensor_double_tree(root_tree_time, root_tree_space),
            sparse_tensor_double_tree(root_tree_time, root_tree_space, 5)
    ]:
        tree = DoubleTree(db_root)
        assert tree.project(0).bfs() == root_tree_time.bfs()
        assert tree.project(1).bfs() == root_tree_space.bfs()


def test_fiber():
    def slow_fiber(i, mu):
        return [
            node.nodes[i] for node in tree.bfs() if node.nodes[not i] is mu
        ]

    for dt_root in [
            full_tensor_double_tree(corner_index_tree(8, 't', 0),
                                    corner_index_tree(8, 'x', 1)),
            sparse_tensor_double_tree(corner_index_tree(8, 't', 0),
                                      corner_index_tree(8, 'x', 1), 8),
            sparse_tensor_double_tree(corner_index_tree(8, 't', 0),
                                      uniform_index_tree(8, 'x'), 8),
            random_double_tree(uniform_index_tree(7, 't'),
                               uniform_index_tree(7, 'x'),
                               7,
                               N=500),
    ]:
        tree = DoubleTree(dt_root)
        for i in [0, 1]:
            for mu in tree.root.bfs(not i):
                assert tree.fiber(i, mu.nodes[not i]).bfs() == slow_fiber(
                    i, mu.nodes[not i])


def test_union():
    """ Test that union indeed copies a tree. """
    time_root = corner_index_tree(7, 't', 0)
    space_root = corner_index_tree(7, 'x', 1)
    from_tree = DoubleTree(full_tensor_double_tree(time_root, space_root))
    to_tree = DoubleTree(DoubleNode((time_root, space_root)))

    assert len(to_tree.bfs()) == 1
    # Copy axis 0 into `to_tree`.
    to_tree.root.union(from_tree.project(0), 0)
    assert len(to_tree.bfs()) == len(time_root.bfs())

    # Copy all subtrees in axis 1 into `to_tree`.
    for item in to_tree.root.bfs(0):
        item.union(from_tree.fiber(1, item.nodes[0]), 1)
    assert len(to_tree.bfs()) == len(from_tree.bfs())

    # Assert double-tree structure is copied as well.
    assert to_tree.root.children[0][0].children[1][0] == \
           to_tree.root.children[1][0].children[0][0]
