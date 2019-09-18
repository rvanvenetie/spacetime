from abc import ABC, abstractmethod
from collections import deque
""" A module for (single-axis) family trees. """


class NodeInterface(ABC):
    """ Represents a node in a family tree: nodes have multiple parents. """
    @abstractmethod
    def is_full(self):
        pass

    @property
    @abstractmethod
    def children(self):
        pass

    @property
    @abstractmethod
    def parents(self):
        pass

    @property
    @abstractmethod
    def marked(self):
        """ A marked field getter/setter.  Useful for bfs/dfs. """
        pass

    @marked.setter
    @abstractmethod
    def marked(self, value):
        pass


class NodeAbstract(NodeInterface):
    """ Partial impl. of NodeInterface, using variables for the properties. """
    children = None
    parents = None
    marked = None

    def __init__(self, parents=None, children=None):
        self.parents = parents if parents else []
        self.children = children if children else []
        self.marked = False


class MetaRoot(NodeAbstract):
    """ Combines the roots of a multi-rooted family tree. """
    def __init__(self, roots):
        if not isinstance(roots, list):
            roots = [roots]
        super().__init__(children=roots)

        # Register self as the parent of the roots. We are now Pater Familias.
        for root in roots:
            assert isinstance(root, NodeAbstract)
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
