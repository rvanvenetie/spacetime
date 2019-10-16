from ..datastructures.tree import MetaRoot
from ..datastructures.tree_view import NodeView, NodeViewInterface, TreeView
from .triangulation import Element2D


class ElementView(NodeView):
    """ View object of an element, storing references to a vertex view. """
    __slots__ = ['vertices_view_idx']

    def refinement_edge(self):
        return (self.vertices_view_idx[1], self.vertices_view_idx[2])

    def newest_vertex(self):
        """ Returns the newest vertex, i.e., vertex 0. """
        return self.vertices_view_idx[0]

    @property
    def area(self):
        return self.node.area


class TriangulationView:
    """ This class represents a (sub)triangulation.

    Currently, this class is initialized by passing it a vertex (sub)tree.
    From this vertex tree, the associated element tree is created. These
    are then stored in tree and `flattened` format.
    """
    def __init__(self, vertex_view):
        """ Initializer given a vertex (sub)tree. """
        if isinstance(vertex_view, NodeViewInterface):
            assert vertex_view.is_metaroot()
        # Store the vertices inside the vertex_view
        self.vertices = vertex_view.bfs()

        # Extract the original element root.
        elem_meta_root = self.vertices[0].node.patch[0].parent
        assert isinstance(elem_meta_root, MetaRoot)
        assert isinstance(elem_meta_root.children[0], Element2D)

        # Mark all necessary vertices -- uses the mark field inside vertex.
        for idx, vertex in enumerate(self.vertices):
            assert not vertex.is_metaroot()
            vertex.node.marked = idx

        # Two helper functions used inside the element tree generation..
        def newest_vertex_in_tree_view(elem):
            """ Does the given element have all vertices in our subtree. """
            return not isinstance(elem.newest_vertex().marked, bool)

        def store_vertices_element_view(elem_view):
            """ Stores the vertex view indices inside the element_view object. """
            if not isinstance(elem_view.node, MetaRoot):
                elem_view.vertices_view_idx = [
                    v.marked for v in elem_view.node.vertices
                ]

        # Now create the associated element tree.
        self.elem_tree_view = TreeView(ElementView(elem_meta_root))
        self.elem_tree_view.deep_refine(
            call_filter=newest_vertex_in_tree_view,
            call_postprocess=store_vertices_element_view)

        # Also store a flattened view of the elements.
        self.elements = self.elem_tree_view.bfs()

        # Create the history object -- uses mark field of the vertex view obj.
        self.history = []
        for elem in self.elements:
            vertex = self.vertices[elem.newest_vertex()]
            if elem.level == 0 or vertex.marked: continue
            vertex.marked = True
            assert len(elem.parents) == 1
            self.history.append((elem.newest_vertex(), elem.parents[0]))

        # Undo marking.
        for vertex in self.vertices:
            vertex.marked = False
            vertex.node.marked = False
