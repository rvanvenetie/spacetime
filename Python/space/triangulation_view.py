from ..datastructures.tree import MetaRoot
from ..datastructures.tree_view import NodeView, NodeViewInterface, TreeView
from .triangulation import Element2D, Vertex


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
        assert self.vertices

        # In the case of stacked views, we must iterate to find the vertices.
        while isinstance(self.vertices[0], NodeViewInterface):
            self.vertices = [v.node for v in self.vertices]
        assert isinstance(self.vertices[0], Vertex)

        # Extract the original element root.
        elem_meta_root = self.vertices[0].patch[0].parent
        assert isinstance(elem_meta_root, MetaRoot)
        assert isinstance(elem_meta_root.children[0], Element2D)

        # Mark all necessary vertices -- uses the mark field inside vertex.
        for idx, vertex in enumerate(self.vertices):
            assert not vertex.is_metaroot()
            vertex.marked = idx

        # Two helper functions used inside the element tree generation..
        def newest_vertex_in_tree_view(elem):
            """ Does the given element have all vertices in our subtree. """
            return not isinstance(elem.newest_vertex().marked, bool)

        def store_vertices_element_view(elem_view):
            """ Store vertex view indices inside the element_view object. """
            if not isinstance(elem_view.node, MetaRoot):
                elem_view.vertices_view_idx = [
                    v.marked for v in elem_view.node.vertices
                ]

        # Now create the associated element tree.
        self.elem_tree_view = TreeView(ElementView(elem_meta_root))
        self.elem_tree_view.deep_refine(
            call_filter=newest_vertex_in_tree_view,
            call_postprocess=store_vertices_element_view)

        # Unmark the vertices
        for vertex in self.vertices:
            vertex.marked = False

        # Also store a flattened list of the elements.
        self.elements = self.elem_tree_view.bfs()
        assert self.elements

        # Create the history object -- uses mark field of the vertex view obj.
        self.history = []
        for elem in self.elements:
            vertex = self.vertices[elem.newest_vertex()]
            if elem.level == 0 or vertex.marked: continue
            vertex.marked = True
            assert len(elem.parents) == 1
            self.history.append((elem.newest_vertex(), elem.parents[0]))

        if not (len(self.history) == len(self.vertices) -
                len(self.vertices[0].parents[0].children)):
            print('Invalid triangulation view object created.')
            print('\thistory = ', self.history)
            print('\tvertices = ', self.vertices)
            print('\troots = ', self.vertices[0].parents[0].children)
            assert False

        # Undo marking.
        for vertex in self.vertices:
            vertex.marked = False
