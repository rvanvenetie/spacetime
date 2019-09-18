from tree import *
import pytest


class Node(NodeABC):
    def __init__(self, parents=None, children=None):
        super().__init__()


def test_ABC():
    with pytest.raises(TypeError):
        root = NodeABC()

    root = Node()
    root.children = [Node(), Node()]
    root.children[0].children = [Node(), Node()]
    assert len(root.bfs()) == 5

    metaroot = MetaRoot([root])
    assert len(metaroot.bfs()) == 5
    assert len(metaroot.bfs(include_metaroots=True)) == 6
