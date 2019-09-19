import random
from collections import defaultdict, deque
from pprint import pprint

import pytest

from .double_tree import DoubleNode, DoubleTree
from .function_test import FakeHaarFunction, FakeOrthoFunction
from .tree import MetaRoot


class FakeMetaRoot(MetaRoot):
    @property
    def level(self):
        return -1

    def is_full(self):
        if not self.roots: return False
        if isinstance(self.roots[0], FakeHaarNode): return len(self.roots) == 1
        if isinstance(self.roots[0], FakeOrthoNode):
            return len(self.roots) == 2
        assert False


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
        return self.nodes[0].level + self.nodes[1].level


def create_roots(node_type, node_class):
    """ Returns a MetaRoot instance, containing all necessary roots. """
    if issubclass(node_class, FakeHaarFunction):
        return FakeMetaRoot(node_class((0, 0), node_type))
    elif issubclass(node_class, FakeOrthoFunction):
        root_0 = node_class((0, 0), node_type)
        root_1 = node_class((0, 1), node_type)
        root_0.nbr = root_1
        root_1.nbr = root_0
        return FakeMetaRoot([root_0, root_1])
    else:
        assert False


def uniform_index_tree(max_level, node_type, node_class=FakeHaarFunction):

    """ Creates a (dummy) index tree.
    
    Creates a field node_type inside the nodes and sets it to the node_type.
    """
    meta_root = create_roots(node_type, node_class)
    meta_root.uniform_refine(max_level)
    return meta_root


def corner_index_tree(max_level,
                      node_type,
                      which_child=0,
                      node_class=FakeHaarFunction):
    """ Creates a (dummy) index tree with 1 element per level.
    
    Creates a field node_type inside the nodes and sets it to the node_type.
    """
    meta_root = create_roots(node_type, node_class)
    Lambda_l = meta_root.roots.copy()
    for _ in range(max_level):
        Lambda_new = []
        for node in Lambda_l:
            children = node.refine()
            Lambda_new.append(children[which_child])
        Lambda_l = Lambda_new
    return meta_root


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


def uniform_full_grid(time_level, space_level, node_class=FakeHaarFunction):
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
    meta_root_haar = uniform_index_tree(5, 't', FakeHaarFunction)
    assert len(meta_root_haar.bfs()) == 2**6 - 1
    root_ortho = uniform_index_tree(5, 't', FakeOrthoFunction)
    assert len(root_ortho.bfs()) == 2**7 - 2


def test_full_tensor():
    DebugDoubleNode.total_counter = 0
    double_root = uniform_full_grid(4, 2, FakeHaarFunction)
    assert DebugDoubleNode.total_counter == (2**5 - 1) * (2**3 - 1)
    DebugDoubleNode.total_counter = 0
    double_root = uniform_full_grid(4, 2, FakeOrthoFunction)
    assert DebugDoubleNode.total_counter == (2**6 - 2) * (2**4 - 2)


def test_sparse_tensor():
    DebugDoubleNode.total_counter = 0
    double_root = uniform_sparse_grid(1, FakeHaarFunction)
    assert DebugDoubleNode.total_counter == 5
    DebugDoubleNode.total_counter = 0
    double_root = uniform_sparse_grid(4, FakeHaarFunction)
    assert DebugDoubleNode.total_counter == 129

    DebugDoubleNode.total_counter = 0
    double_root = uniform_sparse_grid(1, FakeOrthoFunction)
    assert DebugDoubleNode.total_counter == 2 * 2 + 2 * 2 * 4


def test_tree_refine():
    """ Checks that refine only works if all necessary parents are present. """
    meta_root_time = uniform_index_tree(2, 't', FakeHaarFunction)
    meta_root_space = uniform_index_tree(2, 'x', FakeHaarFunction)
    root = DebugDoubleNode((meta_root_time, meta_root_space))
    child, = root.refine(0)
    with pytest.raises(AssertionError):
        # This will violate the double tree constraint.
        child.refine(1)


def test_project():
    for node_cls in [FakeHaarFunction, FakeOrthoFunction]:
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

    for cls in [FakeHaarFunction, FakeOrthoFunction]:
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
    for cls in [FakeHaarFunction, FakeOrthoFunction]:
        meta_root_time = corner_index_tree(7, 't', 0, cls)
        meta_root_space = corner_index_tree(7, 'x', 1, cls)
        from_tree = full_tensor_double_tree(meta_root_time, meta_root_space)
        to_tree = DoubleTree(DoubleNode((meta_root_time, meta_root_space)))

        assert len(to_tree.bfs(include_meta_root=False)) == 0
        assert len(to_tree.bfs(include_meta_root=True)) == 1

        # Copy axis 0 into `to_tree`.
        to_tree.root.union(from_tree.project(0), 0)
        assert len(to_tree.bfs(i=0)) == len(meta_root_time.bfs())

        # Copy all subtrees in axis 1 into `to_tree`.
        for item in to_tree.root.bfs(i=0, include_meta_root=True):
            item.union(from_tree.fiber(1, item.nodes[0]), 1)
        assert len(to_tree.bfs()) == len(from_tree.bfs())

        # Assert double-tree structure is copied as well.
        assert to_tree.root.children[0][0].children[1][0] == \
               to_tree.root.children[1][0].children[0][0]
