from collections import deque
from pprint import pprint

from tree import *


class DummyNode(Node):
    """ Dummy nodes that refines into two children. """

    def __init__(self, labda, node_type, parents=None, children=None):
        super().__init__(labda, parents, children)
        self.node_type = node_type

    def refine(self):
        if self.children: return
        l, n = self.labda
        self.children.append(DummyNode((l + 1, 2 * n), self.node_type, [self]))
        self.children.append(
            DummyNode((l + 1, 2 * n + 1), self.node_type, [self]))
        return self.children

    @property
    def level(self):
        return self.labda[0]

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


def uniform_index_tree(max_level, node_type):
    """ Creates a (dummy) index tree with 2**l elements per level.
    
    Creates a field node_type inside the nodes and sets it to the node_type.
    """
    root = DummyNode((0, 0), node_type)
    Lambda_l = [root]
    for _ in range(max_level):
        Lambda_new = []
        for node in Lambda_l:
            Lambda_new.extend(node.refine())
        Lambda_l = Lambda_new
    return root


def full_tensor_double_tree(level_time, level_space):
    root_tree_time = uniform_index_tree(level_time, 'time')
    root_tree_space = uniform_index_tree(level_space, 'space')

    double_root = DebugDoubleNode(pair(0, root_tree_time, root_tree_space))
    queue = deque()
    queue.append(double_root)
    while queue:
        double_node = queue.popleft()

        for i in [0, 1]:
            if double_node.children[i]: continue
            queue.extend(double_node.refine(i))

    return double_root


def sparse_tensor_double_tree(max_level):
    root_tree_time = uniform_index_tree(max_level, 'time')
    root_tree_space = uniform_index_tree(max_level, 'space')
    double_root = DebugDoubleNode(pair(0, root_tree_time, root_tree_space))
    queue = deque()
    queue.append(double_root)
    while queue:
        double_node = queue.popleft()

        if double_node.level >= max_level: continue
        for i in [0, 1]:
            if double_node.children[i]: continue
            queue.extend(double_node.refine(i))

    return double_root


def test_full_tensor():
    double_root = full_tensor_double_tree(4, 2)
    assert DebugDoubleNode.total_counter == (2**5 - 1) * (2**3 - 1)
    DebugDoubleNode.total_counter = 0


def test_sparse_tensor():
    double_root = sparse_tensor_double_tree(1)
    assert DebugDoubleNode.total_counter == 5
    DebugDoubleNode.total_counter = 0
    double_root = sparse_tensor_double_tree(4)
    assert DebugDoubleNode.total_counter == 129
    DebugDoubleNode.total_counter = 0
