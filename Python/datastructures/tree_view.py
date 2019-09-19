from .tree import NodeAbstract, NodeInterface


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

    def refine(self, make_conforming=False, refine_underlying_tree=False):
        """ Refines the tree view according to the real tree.
        
        Args:
          make_conforming: In case not all parents are present of the children.
          refine_underlying_tree: Refine the underlying tree if necessary."""
        if refine_underlying_tree: self.node.refine()
        for child in self.node.children:
            # Check if we already have this child.
            if child in (n.node for n in self.children): continue

            # Find the parents of the to be created child.
            brothers_view = [
                self.find_brother(parent) for parent in child.parents
            ]

            # We are missing a brother.
            if None in brothers_view:
                if not make_conforming: assert False

                # We want to ensure this brother exists, so recurse.
                for parent in self.parents:
                    parent.refine(make_conforming=True,
                                  refine_underlying_tree=False)
                # Brothers of self should now exist, try again.
                return self.refine(make_conforming, refine_underlying_tree)

            child_view = self.__class__(child, parents=brothers_view)
            for brother_view in brothers_view:
                brother_view.children.append(child_view)
        return self.children
