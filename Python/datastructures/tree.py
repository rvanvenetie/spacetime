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

    @abstractmethod
    def is_full(self):
        pass


class MetaRoot(NodeABC):
    """ Combines the roots of a multi-rooted family tree. """
    def __init__(self, roots):
        if not isinstance(roots, list):
            roots = [roots]
        super().__init__(children=roots)

        # Register self as the parent of the roots. We are now Pater Familias.
        for root in roots:
            assert isinstance(root, NodeABC)
            assert not root.parents
            root.parents = [self]

    def is_full(self):
        return True

    def bfs(self, include_metaroot=False):
        """ Performs a BFS on the family tree rooted at `self`.
        
        Args:
            include_metaroot: whether to return `self` as well.
        """
        queue = deque()
        queue.append(self)
        nodes = []
        while queue:
            node = queue.popleft()
            if node.marked: continue
            nodes.append(node)
            node.marked = True
            queue.extend(node.children)
        for node in nodes:
            node.marked = False
        if not include_metaroot:
            return nodes[1:]
        return nodes
