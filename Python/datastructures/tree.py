from abc import ABC, abstractmethod
from collections import deque
""" A module for (single-axis) family trees. """


class NodeInterface(ABC):
    """ Represents a node in a family tree: nodes have multiple parents. """
    @abstractmethod
    def is_full(self):
        """ Returns whether this node has all possible children present. """
        pass

    @abstractmethod
    def refine(self):
        """ Refines this node to ensure it is full. Returns all children. """
        pass

    @property
    @abstractmethod
    def level(self):
        """ The level of this node. Root has level 0, its children 1, etc. """
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

    def is_leaf(self):
        return len(self.children) == 0

    def _bfs(self):
        """ Performs a BFS on the family tree rooted at `self`.  """
        queue = deque([self])
        nodes = []
        while queue:
            node = queue.popleft()
            if node.marked: continue
            nodes.append(node)
            node.marked = True
            queue.extend(node.children)
        for node in nodes:
            node.marked = False
        return nodes


class NodeAbstract(NodeInterface):
    """ Partial impl. of NodeInterface, using variables for the properties. """
    __slots__ = ['parents', 'children', 'marked']

    def __init__(self, parents=None, children=None):
        self.parents = parents if parents else []
        self.children = children if children else []
        self.marked = False


class BinaryNodeAbstract(NodeAbstract):
    """ Partial impl. of a binary tree node (hence, with a single parent). """
    def __init__(self, parent=None, children=None):
        if parent:
            super().__init__(parents=[parent], children=children)
        else:
            super().__init__(children=children)

    @property
    def parent(self):
        return self.parents[0]

    @parent.setter
    def parent(self, parent):
        assert isinstance(parent, NodeInterface)
        self.parents = [parent]

    def is_full(self):
        return len(self.children) == 2


class MetaRootInterface(NodeInterface):
    def __init__(self, roots):
        """ Registers self as the parent of the roots. """
        # Register self as the parent of the roots. We are now Pater Familias.
        for root in roots:
            assert isinstance(root, NodeAbstract)
            assert not root.parents
            root.parents = [self]

    def is_full(self):
        return True

    @property
    def roots(self):
        """ The roots this MetaRoot is representing (simply the children). """
        return self.children

    @property
    def level(self):
        return -1

    def bfs(self, include_metaroot=False):
        """ Performs a BFS on the family tree rooted at `self`.
        
        Args:
            include_metaroot: whether to return `self` as well.
        """
        nodes = self._bfs()
        if not include_metaroot:
            return nodes[1:]
        else:
            return nodes


class MetaRoot(MetaRootInterface, NodeAbstract):
    """ Combines the roots of a multi-rooted family tree. """
    def __init__(self, roots):
        if not isinstance(roots, list):
            roots = [roots]
        NodeAbstract.__init__(self, children=roots)
        MetaRootInterface.__init__(self, roots=roots)

    def refine(self):
        return self.children

    def uniform_refine(self, max_level):
        """ Ensure that the tree contains up to max_level nodes. """
        nodes = []
        queue = deque(self.roots)
        while queue:
            node = queue.popleft()
            if node.marked: continue
            nodes.append(node)
            node.marked = True
            if node.level < max_level: queue.extend(node.refine())
        for node in nodes:
            node.marked = False
        return nodes

    def __repr__(self):
        return "MR(%s)" % self.roots
