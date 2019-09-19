import pytest

from .tree import *


class Node(NodeAbstract):
    def __init__(self, parents=None, children=None):
        super().__init__()

    def level(self):
        raise NotImplementedError("Cannot call level on fake class.")

    def is_full(self):
        raise NotImplementedError("Cannot call is_full on fake class.")

    def refine(self):
        raise NotImplementedError("Cannot call refine on fake class.")


class BinaryNode(BinaryNodeAbstract):
    def __init__(self, parent=None, children=None):
        super().__init__(parent, children)

    def level(self):
        raise NotImplementedError("Cannot call level on fake class.")

    def refine(self):
        raise NotImplementedError("Cannot call refine on fake class.")


def test_ABC():
    with pytest.raises(TypeError):
        root = NodeAbstract()
    with pytest.raises(TypeError):
        root = NodeInterface()


def test_binary():
    root = BinaryNode()
    root.children = [BinaryNode(root), BinaryNode(root)]
    assert root.children[0].parent == root
    assert root.children[1].parent == root
    assert root.is_full()


def test_bfs():
    root = Node()
    root.children = [Node(), Node()]
    root.children[0].children = [Node(), Node()]
    metaroot = MetaRoot([root])
    assert len(metaroot.bfs()) == 5
    assert len(metaroot.bfs(include_metaroot=True)) == 6
