from abc import ABC, abstractmethod
from collections import deque

from .tree import MetaRoot, MetaRootInterface, NodeAbstract, NodeInterface


class NodeViewInterface(NodeInterface):
    """ This defines a `view` or `subtree` of an existing underlying tree. """

    # Abstract methods.
    @property
    @abstractmethod
    def node(self):
        pass

    # Default implementations of a NodeView interface.
    @property
    def level(self):
        return self.node.level

    def is_full(self):
        """ Do we represent a `full` node? """
        return len(self.children) == len(self.node.children)

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
        if self.is_full(): return self.children
        if children is None: children = self.node.children

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

    def _union(self, other, call_postprocess=None):
        """ Deep-copies the node view tree rooted at `other` into self.

        Args:
          other: Root of the other node view tree that we whish to union. We
                 must have self.node == other.node.
          call_postprocess: This call will be invoked for every pair
              of nodeview objects. First arg will hold a ref to this tree,
              second arg will hold a ref to the second tree.
        """
        if call_postprocess is None: call_postprocess = lambda _, __: None
        assert isinstance(other, NodeViewInterface)
        assert self.node == other.node
        queue = deque([(self, other)])
        my_nodes = []
        while queue:
            my_node, other_node = queue.popleft()
            assert type(my_node) == type(other_node)
            assert my_node.node == other_node.node
            if my_node.marked: continue

            call_postprocess(my_node, other_node)
            my_node.marked = True
            my_nodes.append(my_node)

            my_node.refine(children=[c.node for c in other_node.children])
            assert len(my_node.children) >= len(other_node.children)

            # Only put children that other_node has as well into the queue.
            for my_child in my_node.children:
                for other_child in other_node.children:
                    if my_child.node is other_child.node:
                        queue.append((my_child, other_child))

        # Reset mark field.
        for my_node in my_nodes:
            my_node.marked = False

        return self

    def _deep_refine(self, call_filter=None, call_postprocess=None):
        """ Deep-refines `self` by recursively refining the tree view. 

        Args:
          call_filter: This call determines whether a given node in the 
              argument should be inside the subtree.
          call_postprocess: This call will be invoked with a freshly
              created nodeview object. Can be used to load data, etc.
        """
        if call_filter is None: call_filter = lambda _: True
        if call_postprocess is None: call_postprocess = lambda _: None
        nodes = []
        queue = deque(self.children)
        while queue:
            node = queue.popleft()
            if node.marked: continue
            nodes.append(node)
            call_postprocess(node)
            node.marked = True
            for child in filter(call_filter, node.node.children):
                queue.extend(node.refine(children=[child]))
        for node in nodes:
            node.marked = False

    def __repr__(self):
        return "NV_%s" % self.node


class NodeView(NodeViewInterface, NodeAbstract):
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

    def copy_data_from(self, other):
        """ Copies the appropriate fields from `other` into `self`. """
        assert type(self) == type(other)


class MetaRootView(MetaRootInterface, NodeView):
    def __init__(self, metaroot, node_view_cls=NodeView):
        if isinstance(metaroot, MetaRootView):
            metaroot = metaroot.node
        assert isinstance(metaroot, MetaRoot)
        assert issubclass(node_view_cls, NodeView)

        # Create a nodeview object for the roots.
        roots = [node_view_cls(node=rt) for rt in metaroot.roots]

        # Initialize the underlying objects.
        NodeView.__init__(self, node=metaroot, children=roots)
        MetaRootInterface.__init__(self)

        # Register self as the parent of the roots.
        for root in roots:
            assert not root.parents
            root.parents = [self]

    def union(self, other, call_postprocess=None):
        """ Deep-copies the MetaRootView tree rooted at `other` into self. """
        return self._union(other, call_postprocess)

    def deep_copy(self):
        """ Deep-copies `self` into a new NodeView tree. """
        def callback(new_node, my_node):
            return new_node.copy_data_from(my_node)

        new_metaroot = self.__class__(self,
                                      node_view_cls=self.roots[0].__class__)
        return new_metaroot.union(self, call_postprocess=callback)

    def deep_refine(self, call_filter=None, call_postprocess=None):
        """ Deep-refines `self` by recursively refining the tree view. """
        self._deep_refine(call_filter, call_postprocess)

    def uniform_refine(self, max_level):
        self._deep_refine(call_filter=lambda n: n.level <= max_level)

    def __iadd__(self, other):
        def callback(my_node, other_node):
            if not isinstance(my_node, MetaRootInterface):
                my_node += other_node

        return self.union(other, call_postprocess=callback)

    def __imul__(self, x):
        assert isinstance(x, (int, float, complex)) and not isinstance(x, bool)
        for node in self.bfs():
            node *= x
        return self

    def __repr__(self):
        return "MRV(%s)" % self.roots
