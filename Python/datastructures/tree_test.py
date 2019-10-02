import pytest

from .function_test import FakeHaarFunction, FakeOrthoFunction
from .tree import *


class FakeMetaRoot(MetaRoot):
    @property
    def level(self):
        return -1

    def is_full(self):
        if not self.roots: return False
        if isinstance(self.roots[0], FakeHaarFunction):
            return len(self.roots) == 1
        if isinstance(self.roots[0], FakeOrthoFunction):
            return len(self.roots) == 2
        assert False


class FakeNode(NodeAbstract):
    def __init__(self, parents=None, children=None):
        super().__init__()

    @property
    def level(self):
        raise NotImplementedError("Cannot call level on fake class.")

    def is_full(self):
        raise NotImplementedError("Cannot call is_full on fake class.")

    def refine(self):
        raise NotImplementedError("Cannot call refine on fake class.")


class FakeBinaryNode(BinaryNodeAbstract):
    def __init__(self, parent=None, children=None):
        super().__init__(parent, children)

    @property
    def level(self):
        raise NotImplementedError("Cannot call level on fake class.")

    def refine(self):
        raise NotImplementedError("Cannot call refine on fake class.")


def create_roots(node_type, node_class):
    """ Returns a MetaRoot instance, containing all necessary roots. """
    if issubclass(node_class, FakeHaarFunction):
        return FakeMetaRoot(node_class((0, 0), node_type))
    elif issubclass(node_class, FakeOrthoFunction):
        root_0 = node_class((0, 0), node_type)
        root_1 = node_class((0, 1), node_type)
        root_0.nbr = root_1
        root_1.nbr = root_0
        return FakeMetaRoot([root_0, root_1])
    else:
        assert False


def uniform_index_tree(max_level, node_type, node_class=FakeHaarFunction):
    """ Creates a (dummy) index tree.
    
    Creates a field node_type inside the nodes and sets it to the node_type.
    """
    meta_root = create_roots(node_type, node_class)
    meta_root.uniform_refine(max_level)
    return meta_root


def corner_index_tree(max_level,
                      node_type,
                      which_child=0,
                      node_class=FakeHaarFunction):
    """ Creates a (dummy) index tree with 1 element per level.
    
    Creates a field node_type inside the nodes and sets it to the node_type.
    """
    meta_root = create_roots(node_type, node_class)
    Lambda_l = meta_root.roots.copy()
    for _ in range(max_level):
        Lambda_new = []
        for node in Lambda_l:
            children = node.refine()
            Lambda_new.append(children[which_child])
        Lambda_l = Lambda_new
    return meta_root


def test_ABC():
    with pytest.raises(TypeError):
        root = NodeAbstract()
    with pytest.raises(TypeError):
        root = NodeInterface()


def test_binary():
    root = FakeBinaryNode()
    root.children = [FakeBinaryNode(root), FakeBinaryNode(root)]
    assert root.children[0].parent == root
    assert root.children[1].parent == root
    assert root.is_full()


def test_bfs():
    root = FakeNode()
    root.children = [FakeNode(), FakeNode()]
    root.children[0].children = [FakeNode(), FakeNode()]
    metaroot = MetaRoot([root])
    assert len(metaroot.bfs()) == 5
    assert len(metaroot.bfs(include_metaroot=True)) == 6


def test_diamond_structure():
    root = FakeNode()
    root.children = [FakeNode(), FakeNode()]
    root.children[0].children = [FakeNode()]
    root.children[0].parents.append(root.children[1])
    metaroot = MetaRoot([root])
    assert len(metaroot.bfs()) == 4


def test_uniform_index_tree():
    meta_root_haar = uniform_index_tree(5, 't', FakeHaarFunction)
    assert len(meta_root_haar.bfs()) == 2**6 - 1
    root_ortho = uniform_index_tree(5, 't', FakeOrthoFunction)
    assert len(root_ortho.bfs()) == 2**7 - 2
