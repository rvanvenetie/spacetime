from collections import defaultdict, deque


class Node(object):
    """ Represents a node in a single coordinate. """

    def __init__(self, labda, parents=None, children=None):
        self.labda = labda
        self.parents = parents if parents else []
        self.children = children if children else []

        # Create a marked field; useful for bfs/dfs.
        self.marked = False

    @property
    def support(self):
        pass

    def __hash__(self):
        return hash(self.labda)


def pair(i, item_i, item_not_i):
    """ Helper function to create a pair.
    
    Given coordinate i, returns a pair with item_i in coordinate i,
    and item_not_i in coordinate not i.
    """
    result = [None, None]
    result[i] = item_i
    result[not i] = item_not_i
    return tuple(result)


class DoubleNode(object):
    def __init__(self, nodes, parents=None, children=None):
        """ Creates double node, with nodes pointing to `single` index node. """
        self.nodes = tuple(nodes)
        self.parents = parents if parents else ([], [])
        self.children = children if children else ([], [])
        assert len(self.parents) == 2 and len(
            self.children) == 2 and len(nodes) == 2

        # Create a marked field useful for bfs/dfs.
        self.marked = False

    def is_leaf(self):
        return len(self.children[0]) == 0 and len(self.children[1]) == 0

    def find_ghost_child(self, i, nodes):
        """ Checks if the `to-be-created` child already exists.

        Arguments:
            i: the axis along which you are trying to refine the current node.
            nodes: tuple of nodes that would form the new DoubleNode.
        Returns:
            The DoubleNode if such a child exists, if not it simply None."""
        for parent_not_i in self.parents[not i]:
            assert len(parent_not_i.children[i]) > 0
            for children_i in parent_not_i.children[i]:
                for children_not_i in children_i.children[not i]:
                    if children_not_i.nodes == nodes:
                        return children_not_i
        return None

    def find_brother(self, i, nodes):
        """ Finds the given brother in the given axis. """
        for parent_i in self.parents[i]:
            for sibling_i in parent_i.children[i]:
                if sibling_i.nodes == nodes:
                    return sibling_i
        return None

    def remove_me(self):
        assert self.is_leaf()
        for i in [0, 1]:
            for parent in self.parents[i]:
                parent.children[i].remove(self)

    def refine(self, i):
        """ Refines the node in the `i`-th coordinate.
        
        If the node that will be introduced has multiple parents, then these
        must be brothers of self (and already exist).
        """
        if self.children[i]: return self.children[i]

        for child_i in self.nodes[i].children:
            child_nodes = pair(i, child_i, self.nodes[not i])
            ghost_child = self.find_ghost_child(i, child_nodes)

            if ghost_child:
                # Child node already exists, update child/parent relations.
                ghost_child.parents[i].append(self)
                self.children[i].append(ghost_child)
            else:

                # Collect all parents (brothers) necessary to refine the child.
                if len(child_i.parents) == 1:
                    parents = [self]
                else:
                    parents = []
                    for brother_nodes in (pair(i, parent, self.nodes[not i])
                                          for parent in child_i.parents):
                        tmp = self.find_brother(i, brother_nodes)
                        # Ensure that we have found this!
                        assert tmp
                        parents.append(tmp)

                child = self.__class__(child_nodes)

                # Update the parent/child relations
                for parent in parents:
                    child.parents[i].append(parent)
                    parent.children[i].append(child)

        return self.children[i]

    def __repr__(self):
        return "{} x {}".format(self.nodes[0], self.nodes[1])


class DoubleTree(object):
    def __init__(self, root):
        self.root = root

        self.fibers = [{}, {}]
        for i in [0, 1]:
            for node in bfs(self.root, i):
                self.fibers[not i][node.nodes[i]] = [
                    n.nodes[not i] for n in bfs(node, not i)
                ]

    def project(self, i):
        """ Return the tree of single nodes in axis i. """
        return self.double_root.nodes[i]

    def fiber(self, i, mu):
        """ Return the fiber of double-node mu in axis i.
        
        The fiber is the tree of single-nodes in axis i frozen at coordinate mu
        in the other axis. """
        return self.fibers[i][mu]


def bfs(roots, i=None):
    """ Does a bfs for the given roots.
    
    If `i` is set, we are traversing a double tree, for coordinate i.
    """
    if not isinstance(roots, list): roots = [roots]
    queue = deque()
    queue.extend(roots)
    nodes = []
    while queue:
        node = queue.popleft()
        if node.marked: continue
        nodes.append(node)
        node.marked = True
        # Add the children to the queue.
        if i is not None:
            queue.extend(node.children[i])
        else:
            queue.extend(node.children[0])
            queue.extend(node.children[1])

    for node in nodes:
        node.marked = False
    return nodes
