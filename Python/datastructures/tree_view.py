from .tree import NodeAbstract


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

    def refine(self, refine_underlying_tree=False):
        """ Refines the tree view according to the real tree.
        
        If refine_underlying_tree is true, this will refine the actual tree
        if necessary. """
        if refine_underlying_tree: self.node.refine()
        for child in self.node.children:
            # Check if we already have this child.
            if child in (n.node for n in self.children): continue

            # Find the parents of the to be created child.
            brothers_view = [
                self.find_brother(parent) for parent in child.parents
            ]
            assert None not in brothers_view
            child_view = self.__class__(child, parents=brothers_view)
            for brother_view in brothers_view:
                brother_view.children.append(child_view)
        return self.children
