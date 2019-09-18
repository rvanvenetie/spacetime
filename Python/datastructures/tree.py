from abc import ABC, abstractmethod
from collections import deque
""" A module for (single-axis) family trees. """


class NodeABC(ABC):
    """ Represents a node in a family tree: nodes have multiple parents. """
    @abstractmethod
    def __init__(self, parents=None, children=None):
        self.parents = parents if parents else []
        self.children = children if children else []

        # Create a marked field -- useful for bfs/dfs.
        self.marked = False

    def bfs(self):
        return _bfs(self)


class MetaRoot(NodeABC):
    """ Combines the roots of a multi-rooted family tree. """
    def __init__(self, roots):
        assert isinstance(roots, list)
        super().__init__(children=roots)

        # Register is as the parent of the roots. We are now Pater Familias.
        for root in roots:
            assert isinstance(root, NodeABC)
            assert not root.parents
            root.parents = [self]

    def bfs(self, include_metaroots=False):
        return _bfs(self, include_metaroots=include_metaroots)


def _bfs(root, include_metaroots=False):
    """ Performs a BFS on the family tree rooted at `root`.
    
    Args:
        root (NodeABC): the root;
        include_metaroots: whether to return MetaRoot nodes as well.
    """
    queue = deque()
    queue.append(root)
    nodes = []
    while queue:
        node = queue.popleft()
        if node.marked: continue
        nodes.append(node)
        node.marked = True
        queue.extend(node.children)
    for node in nodes:
        node.marked = False
    if not include_metaroots:
        nodes = [node for node in nodes if not isinstance(node, MetaRoot)]
    return nodes
