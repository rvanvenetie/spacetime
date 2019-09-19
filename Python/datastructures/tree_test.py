import pytest

from tree import *


class Node(NodeAbstract):
    def __init__(self, parents=None, children=None):
        super().__init__()

    def is_full(self):
        raise NotImplementedError("Cannot call method on fake class.")


def test_ABC():
    with pytest.raises(TypeError):
        root = NodeAbstract()
    with pytest.raises(TypeError):
        root = NodeInterface()


def test_bfs():
    root = Node()
    root.children = [Node(), Node()]
    root.children[0].children = [Node(), Node()]
    metaroot = MetaRoot([root])
    assert len(metaroot.bfs()) == 5
    assert len(metaroot.bfs(include_metaroot=True)) == 6
