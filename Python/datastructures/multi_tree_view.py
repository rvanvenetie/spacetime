from abc import ABC, abstractmethod
from collections import defaultdict, deque
from operator import eq

from .tree import MetaRootInterface, NodeInterface


def _replace(i, items, item_i):
    """ Helper function to replace a single item in a list.  """
    result = list(items)
    result[i] = item_i
    return result


class MultiNodeViewInterface(NodeInterface):
    """ Class that represents a multinode interface. """
    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def nodes(self):
        pass

    @property
    def level(self):
        return sum(n.level for n in self.nodes)

    def is_leaf(self):
        return all(len(self.children[i]) == 0 for i in range(self.dim))

    def is_full(self, i=None):
        if i is None:
            return all(self.is_full(i) for i in range(self.dim))
        else:
            return len(self.children[i]) == len(self.nodes[i].children)

    def coarsen(self):
        return self._coarsen()

    def refine(self,
               i=None,
               children=None,
               call_filter=None,
               make_conforming=False):
        if i is None and self.dim == 1: i = 0
        return self._refine(i=i,
                            children=children,
                            call_filter=call_filter,
                            make_conforming=make_conforming)

    def _sparse_refine(self, max_level):
        """ Refines this multi tree to a sparse grid multitree. """
        self._deep_refine(call_filter=lambda n: n.level <= max_level)

    def _uniform_refine(self, max_levels=None):
        """ Uniformly refine the multi tree rooted at `self`. """
        call_filter = None
        if isinstance(max_levels, int):
            max_levels = [max_levels] * self.dim
        if max_levels:
            call_filter = lambda n: all(n[i].level <= max_levels[i]
                                        for i in range(self.dim))
        self._deep_refine(call_filter=call_filter)

    # Real implementations follow below.
    def _refine(self,
                i,
                children=None,
                call_filter=None,
                make_conforming=False):
        """ Refines the node in the `i`-th coordinate.
        
        If the node that will be introduced has multiple parents, then these
        must be brothers of self (and already exist).

        Args:
          i: The axis we are considering.
          children: If set, the list of children to create. If none, refine
                    all children that exist in the underlying tree.
          call_filter: This function can be used to filter children. It is 
            called with the multi-node that is to be created.
        """
        if self.is_full(i): return self.children[i]
        if children is None: children = self.nodes[i].children
        if call_filter is None: call_filter = lambda _: True

        for child_i in children:
            child_nodes = _replace(i, self.nodes, child_i)

            # Skip if this child has already exists, or if we don't pass the filter.
            if child_nodes in (
                    n.nodes
                    for n in self.children[i]) or not call_filter(child_nodes):
                continue

            # Ensure all the parents of the to be created child exist.
            # These are brothers in various axis of the current node.
            brothers = []
            for j in range(self.dim):
                brothers.append([
                    self._find_brother(
                        _replace(j, child_nodes, child_parent_j), j, i,
                        make_conforming)
                    for child_parent_j in child_nodes[j].parents
                ])

            child = self.__class__(nodes=child_nodes, parents=brothers)
            for j in range(self.dim):
                for brother in brothers[j]:
                    brother.children[j].append(child)

        return self.children[i]

    def _coarsen(self):
        """ Removes `self' from the double tree. """
        assert self.is_leaf()
        for i in range(self.dim):
            for parent in self.parents[i]:
                parent.children[i].remove(self)

    def _find_brother(self, nodes, j, i, make_conforming=False):
        """ Finds the given brother in the given axes.
        
        A brother shares a parent with self in the `i`-axis. That is,
        if self has a parent in the `j`-axis, then the brother would 
        have the same parent in the `i`-axis.
        """
        if self.nodes == nodes: return self
        for parent_j in self.parents[j]:
            for sibling_i in parent_j.children[i]:
                if sibling_i.nodes == nodes:
                    return sibling_i

        # We didn't find the brother, lets create it.
        assert make_conforming
        for parent_j in self.parents[j]:
            parent_j.refine(i, children=[nodes[i]], make_conforming=True)

        return self._find_brother(nodes, j, i)

    def _union(self, other, call_filter=None, call_postprocess=None):
        """ Deep-copies the node view tree rooted at `other` into self.

        Args:
          other: Root of the other node view tree that we whish to union. We
                 must have self.node == other.node.
          call_filter: This call determines whether a given node of the 
              other tree should be inside this subtree.
          call_postprocess: This call will be invoked for every pair
              of nodeview objects. First arg will hold a ref to this tree,
              second arg will hold a ref to the second tree.
        """
        if call_filter is None: call_filter = lambda _: True
        if call_postprocess is None: call_postprocess = lambda _, __: None
        assert isinstance(other, MultiNodeViewInterface)
        assert self.nodes == other.nodes and self.dim == other.dim
        queue = deque([(self, other)])
        my_nodes = []
        while queue:
            my_node, other_node = queue.popleft()
            assert my_node.nodes == other_node.nodes
            if my_node.marked: continue

            call_postprocess(my_node, other_node)
            my_node.marked = True
            my_nodes.append(my_node)

            for i in range(self.dim):
                # Refine according to the call_filter.
                my_node._refine(
                    i=i,
                    children=[c.nodes[i] for c in other_node.children[i]],
                    call_filter=call_filter)

                # Only put children that other_node has as well into the queue.
                for my_child in my_node.children[i]:
                    for other_child in other_node.children[i]:
                        if my_child.nodes == other_child.nodes:
                            queue.append((my_child, other_child))

        # Reset mark field.
        for my_node in my_nodes:
            my_node.marked = False

    def _deep_refine(self, call_filter=None, call_postprocess=None):
        """ Deep-refines `self` by recursively refining the multitree. 

        Args:
          call_filter: This call determines whether a given multinode 
            should be inside the subtree.
          call_postprocess: This call will be invoked with a freshly
              created multinode object. Can be used to load data, etc.
        """
        if call_filter is None: call_filter = lambda _: True
        if call_postprocess is None: call_postprocess = lambda _: None
        my_nodes = []
        queue = deque([self])
        while queue:
            my_node = queue.popleft()
            if my_node.marked: continue
            my_nodes.append(my_node)
            call_postprocess(my_node)
            my_node.marked = True
            for i in range(self.dim):
                my_node._refine(i,
                                call_filter=call_filter,
                                make_conforming=True)
                queue.extend(my_node.children[i])

        for my_node in my_nodes:
            my_node.marked = False

    def _bfs(self):
        """ Performs a BFS on the multi tree rooted at `self`.  """
        queue = deque([self])
        nodes = []
        while queue:
            node = queue.popleft()
            if node.marked: continue
            nodes.append(node)
            node.marked = True
            for i in range(self.dim):
                queue.extend(node.children[i])
        for node in nodes:
            node.marked = False
        return nodes

    def __repr__(self):
        return ' x '.join(map(str, self.nodes))


class MultiNodeView(MultiNodeViewInterface):
    __slots__ = ['nodes', 'parents', 'children', 'marked']

    def __init__(self, nodes, parents=None, children=None):
        """ Creates multi node, with nodes pointing to `single` index node. """
        self.nodes = list(nodes)
        self.parents = parents if parents else [[] for _ in range(self.dim)]
        self.children = children if children else [[] for _ in range(self.dim)]
        assert len(self.parents) == self.dim and len(
            self.children) == self.dim and len(nodes) == self.dim

        # Create a marked field useful for bfs/dfs.
        self.marked = False


class MultiTree:
    """ Class that holds the root of the tree. """
    __slots__ = ['root']

    def __init__(self, root):
        self.root = root

    def bfs(self, include_meta_root=False):
        """ Does a bfs from the root.
        
        Args:
            include_meta_root: if false, filter out all MetaRoot nodes.
        """
        nodes = self.root._bfs()
        if not include_meta_root:
            nodes = [
                n for n in nodes if not any(
                    isinstance(n.nodes[j], MetaRootInterface)
                    for j in range(self.root.dim))
            ]
        return nodes

    def deep_copy(self,
                  mlt_node_cls=None,
                  mlt_tree_cls=None,
                  call_postprocess=None):
        """ Copies the current multitree. """
        if mlt_node_cls is None: mlt_node_cls = self.root.__class__
        if mlt_tree_cls is None: mlt_tree_cls = self.__class__
        multi_root = mlt_node_cls(self.root.nodes)
        multi_root._union(self.root, call_postprocess=call_postprocess)
        multi_tree = mlt_tree_cls(multi_root)
        return multi_tree
