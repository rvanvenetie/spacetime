from collections import deque

from .tree import MetaRoot, NodeAbstract, NodeInterface


class NodeView(NodeAbstract):
    """ This defines a `view` or `subtree` of an existing underlying tree. """
    __slots__ = ['node']

    def __init__(self, node, parents=None, children=None):
        """ Initializes this node `view`.
        
        Args:
          node: the real underlying node we are associated with.
          parents: the parents of this node view object.
          children: the children of this node view object.
        """
        assert isinstance(node, NodeInterface)
        super().__init__(parents=parents, children=children)
        self.node = node

    def is_full(self):
        """ Do we represent a `full` node? """
        return not self.children or len(self.children) == len(
            self.node.children)

    def find_brother(self, node):
        """ Finds a brother of this node. """
        if node == self.node: return self
        for parent in self.parents:
            for sibling in parent.children:
                if sibling.node == node:
                    return sibling
        return None

    def refine(self, children=None, make_conforming=True):
        """ Refines all children in the tree view according to the real tree.
        
        Args:
          children: If set, the list of children to create. If none, refine
                    all children that exist in the underlying tree.
          make_conforming: Ensure that the tree constraint is maintained.
        """
        if children is None:
            children = self.node.children

        for child in children:
            # If this child does not exist in underlying tree, we can stop.
            if child not in self.node.children: continue

            # Check if we already have this child.
            if child in (n.node for n in self.children): continue

            # Find the parents of the to be created child.
            brothers_view = []
            for brother in child.parents:
                brother_view = self.find_brother(brother)

                # If a parent of the to be created child is missing, then
                # create it if `make_conforming` is set to true.
                if brother_view is None:
                    if not make_conforming: assert False
                    # Create this missing brother.
                    for parent_view in self.parents:
                        parent_view.refine(children=[brother],
                                           make_conforming=True)
                    brother_view = self.find_brother(brother)

                # The brother_view must now exist.
                assert brother_view
                brothers_view.append(brother_view)

            child_view = self.__class__(child, parents=brothers_view)
            for brother_view in brothers_view:
                brother_view.children.append(child_view)

        return self.children

    def copy_data_from(self, other):
        """ Copies the appropriate fields from `other` into `self`. """
        assert type(self) == type(other)


class MetaRootView(MetaRoot):
    def deep_copy(self):
        """ Deep-copies `self` into a new NodeView tree. """
        def callback(new_node, my_node):
            return new_node.copy_data_from(my_node)

        new_roots = []
        for root in self.roots:
            new_root = root.__class__(root.node)
            new_roots.append(new_root)
        other = self.__class__(roots=new_roots)
        return other.union(self, callback=callback)

    def union(self, other, callback):
        assert isinstance(other, MetaRootView)
        queue = deque()
        queue.extend(zip(self.roots, other.roots))
        nodes = []
        while queue:
            my_node, other_node = queue.popleft()
            assert type(my_node) == type(other_node)
            assert my_node.node == other_node.node
            if my_node.marked: continue

            callback(my_node, other_node)
            my_node.marked = True
            nodes.append(my_node)

            if other_node.children:
                my_node.refine(children=[c.node for c in other_node.children])
            assert len(my_node.children) >= len(other_node.children)

            # Hidden "quadratic" loop, RIP..
            for other_child in other_node.children:
                for my_child in my_node.children:
                    if other_child.node is my_child.node:
                        queue.append((my_child, other_child))
        for node in nodes:
            node.marked = False
        return self

    def __iadd__(self, other):
        def callback(my_node, other_node):
            my_node += other_node

        return self.union(other, callback=callback)

    def __imul__(self, number):
        assert isinstance(number, float)
        for node in self.bfs():
            node *= number
        return self

    def __repr__(self):
        return "MRV(%s)" % self.roots


class NodeVector(NodeView):
    """ This is a vector on a subtree of an existing underlying tree. """
    def __init__(self, node, value=0.0, parents=None, children=None):
        assert isinstance(node, NodeInterface)
        super().__init__(node, parents=parents, children=children)
        self.value = value

    @property
    def level(self):
        return self.node.level

    def copy_data_from(self, other):
        super().copy_data_from(other)
        self.value = other.value

    def __iadd__(self, other):
        """ Shallow `add` operator. """
        self.value += other.value
        return self

    def __imul__(self, number):
        self.value *= number
        return self
