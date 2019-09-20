from collections import deque

from .tree import MetaRoot, NodeAbstract, NodeInterface


class MetaRootView(MetaRoot):
    def deep_copy(self):
        """ Deep-copies `self` into a new NodeView tree. """
        other = self.__class__(self.node, parents=None, children=None)
        queue = deque()
        queue.append((other, self))
        nodes = []
        while queue:
            new_node, my_node = queue.popleft()
            assert new_node.node == my_node.node
            new_node.shallow_copy_from(my_node)

            if my_node.marked: continue
            my_node.marked = True
            nodes.append(my_node)

            if my_node.children:
                print(my_node, my_node.node,
                      [c.node for c in my_node.children])
                new_children = new_node.refine(
                    children=[c.node for c in my_node.children])
                print("hier")
                assert len(new_children) == len(my_node.children)
                queue.extend(zip(new_children, my_node.children))
        for node in nodes:
            node.marked = False

        return other


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

    def refine_until_complete(self):
        """ Refine until we contain all nodes of the underlying tree. """
        to_refine = deque()
        to_refine.append(self)
        while to_refine:
            node = to_refine.popleft()
            if node.is_leaf() or not node.is_full():
                print("refining node ", node.node, node.node.children)
                to_refine.extend(node.refine())

    def shallow_copy_from(self, other):
        """ Copies the appropriate fields from `other` into `self`. """
        assert type(self) == type(other)
        self.node = other.node


class NodeVector(NodeView):
    """ This is a vector on a subtree of an existing underlying tree. """
    def __init__(self, node, value=0.0, parents=None, children=None):
        assert isinstance(node, NodeInterface)
        super().__init__(node, parents=parents, children=children)
        self.value = value

    def shallow_copy_from(self, other):
        super().shallow_copy_from(other)
        self.value = other.value

    def __iadd__(self, other):
        """ TODO """
        assert isinstance(other, NodeVector)
        queue = deque()
        queue.append((self, other))
        nodes = []
        while queue:
            my_node, other_node = queue.popleft()
            assert my_node.node == other_node.node
            if my_child.marked: continue

            my_node.value += other_node.value
            my_child.marked = True
            nodes.append(my_node)

            if other_node.children:
                my_node.refine(children=[c.node for c in other_node.children])

            # Hidden "quadratic" loop, RIP..
            for other_child in other_node.children:
                for my_child in my_node.children:
                    if other_child.node == my_child.node:
                        queue.append((my_child, other_child))
        for node in nodes:
            node.marked = False

    def __eq__(self, other):
        return isinstance(
            other, NodeVector
        ) and self.node == other.node and self.value == other.value
