from collections import defaultdict, deque


class Node:
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

    def bfs(self):
        return bfs(self)


def pair(i, item_i, item_not_i):
    """ Helper function to create a pair.
    
    Given coordinate i, returns a pair with item_i in coordinate i,
    and item_not_i in coordinate not i.
    """
    result = [None, None]
    result[i] = item_i
    result[not i] = item_not_i
    return tuple(result)


class DoubleNode:
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

    def find_brother(self, i, nodes):
        """ Finds the given brother in the given axis.
        
        A brother shares a parent with self in the `i`-axis.
        """
        if self.nodes == nodes: return self
        for parent_i in self.parents[i]:
            for sibling_i in parent_i.children[i]:
                if sibling_i.nodes == nodes:
                    return sibling_i
        return None

    def find_step_brother(self, i, nodes):
        """ Finds a step brother in the given axis.

        A step brother shares a parent with self in the `other`-axis. That is,
        if self has a parent in the `i`-axis, then the step brother would have
        this parent in the `not i`-axis.
        """
        if self.nodes == nodes: return self
        for parent_i in self.parents[i]:
            for sibling_not_i in parent_i.children[not i]:
                if sibling_not_i.nodes == nodes:
                    return sibling_not_i
        return None

    def refine(self, i):
        """ Refines the node in the `i`-th coordinate.
        
        If the node that will be introduced has multiple parents, then these
        must be brothers of self (and already exist).
        """
        for child_i in self.nodes[i].children:
            child_nodes = pair(i, child_i, self.nodes[not i])

            # Skip if this child has already exists.
            if child_nodes in (n.nodes for n in self.children[i]):
                continue

            # Ensure all the parents of the to be created child exist.
            # These are brothers, in i and not i axis, of the current node.
            brothers = [
                self.find_brother(i, pair(i, child_parent_i,
                                          child_nodes[not i]))
                for child_parent_i in child_nodes[i].parents
            ]
            step_brothers = [
                self.find_step_brother(
                    not i, pair(i, child_nodes[i], child_parent_not_i))
                for child_parent_not_i in child_nodes[not i].parents
            ]

            # TODO: Instead of asserting, we could create the (step)brothers.
            assert None not in brothers
            assert None not in step_brothers

            child = self.__class__(child_nodes)
            for brother in brothers:
                child.parents[i].append(brother)
                brother.children[i].append(child)
            for step_brother in step_brothers:
                child.parents[not i].append(step_brother)
                step_brother.children[not i].append(child)

        return self.children[i]

    def coarsen(self):
        """ Removes `self' from the double tree. """
        assert self.is_leaf()
        for i in [0, 1]:
            for parent in self.parents[i]:
                parent.children[i].remove(self)

    def bfs(self, i):
        return bfs(self, i)

    def __repr__(self):
        return "{} x {}".format(self.nodes[0], self.nodes[1])


class DoubleTree:
    def __init__(self, root):
        self.root = root

        self.fibers = ({}, {})
        for i in [0, 1]:
            for node in self.root.bfs(i):
                self.fibers[not i][node.nodes[i]] = [
                    n.nodes[not i] for n in node.bfs(not i)
                ]

    def project(self, i):
        """ Return the list of single nodes in axis i. """
        return self.root.nodes[i].bfs()

    def fiber(self, i, mu):
        """ Return the fiber of double-node mu in axis i.
        
        The fiber is the tree of single-nodes in axis i frozen at coordinate mu
        in the other axis. """
        return self.fibers[i][mu]

    def bfs(self):
        return bfs(self.root)


def bfs(root, i=None):
    """ Does a bfs from the given single node or double node. """
    queue = deque()
    queue.append(root)
    nodes = []
    while queue:
        node = queue.popleft()
        if node.marked: continue
        nodes.append(node)
        node.marked = True
        # Add the children to the queue.
        if isinstance(node, Node):
            assert i is None
            queue.extend(node.children)
        else:
            if i is not None:
                queue.extend(node.children[i])
            else:
                queue.extend(node.children[0])
                queue.extend(node.children[1])
    for node in nodes:
        node.marked = False
    return nodes
