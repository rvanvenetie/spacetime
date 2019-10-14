import itertools

import numpy as np

from ..datastructures.tree import BinaryNodeAbstract, MetaRoot, NodeAbstract


class Vertex(NodeAbstract):
    """ A vertex in a locally refined triangulation.
    
    Vertices also form a (family)tree, induced by the NVB-relation.

    Args:
      level: the uniform level that introduces this vertex. 
      x,y: the physical coordinates
      on_domain_boundary: does this vertex lie on the domain boundary?
      patch: the elements that surround this vertex
      parents: the NVB-parents
      children: the child nodes that were induces by this vertex
    """
    __slots__ = ['level', 'x', 'y', 'on_domain_boundary', 'patch']

    def __init__(self,
                 level,
                 x,
                 y,
                 on_domain_boundary,
                 patch=None,
                 parents=None,
                 children=None):
        super().__init__(parents=parents, children=children)
        self.level = level
        self.x = x
        self.y = y
        self.on_domain_boundary = on_domain_boundary
        self.patch = patch if patch else []

        # Sanity check.
        assert (all([p.level == self.level - 1 for p in self.parents]))

    def refine(self):
        if not self.is_full():
            for elem in self.patch:
                elem.refine()
        return self.children

    def is_full(self):
        return all(elem.is_full() for elem in self.patch)

    def as_array(self):
        return np.array([self.x, self.y], dtype=float)

    def __repr__(self):
        return 'V({}, [{}, {}])'.format(self.level, self.x, self.y)


class Element2D(BinaryNodeAbstract):
    """ A element as part of a locally refined triangulation. """
    __slots__ = ['level', 'vertices', 'area']

    def __init__(self, level, vertices, parent=None):
        """ Instantiates the element object.

        Arguments:
            level: uniform level that introduces this element
            vertices: array of three Vertex references. 
            parent: reference to the parent of this element.
            """
        super().__init__(parent=parent)
        self.level = level
        self.vertices = vertices
        self.children = []  # References to the children of this element.

        # Indices of my neighbours, ordered by the edge opposite vertex i.
        self.neighbours = [None, None, None]
        if parent:
            self.area = parent.area / 2
            assert parent.level + 1 == self.level

    def refine(self):
        """ Refines the trianglulation so that element `elem` is bisected.

        If this triangle was already bisected, then this function has no effect.
        """
        if not self.is_full():
            nbr = self.neighbours[0]
            if not nbr:  # Refinement edge of `elem` is on domain boundary.
                self._bisect()
            # Shared refinement edge.
            elif nbr.edge(0) != self.reversed_edge(0):
                nbr.refine()
                return self.refine()
            else:
                self._bisect_with_nbr()
        return self.children

    def newest_vertex(self):
        """ Returns the newest vertex, i.e., vertex 0. """
        return self.vertices[0]

    def edge(self, i):
        assert 0 <= i <= 2
        return [self.vertices[(i + 1) % 3], self.vertices[(i + 2) % 3]]

    def reversed_edge(self, i):
        assert 0 <= i <= 2
        return [self.vertices[(i + 2) % 3], self.vertices[(i + 1) % 3]]

    def is_leaf(self):
        return not len(self.children)

    def to_barycentric_coordinates(self, p):
        """ Returns the barycentric coordinates for a point p. """

        # https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
        a, b, c = [self.vertices[i].as_array() for i in range(3)]
        v0 = b - a
        v1 = c - a
        v2 = np.array(p) - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = (d00 * d11 - d01 * d01)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom

        return np.array([1 - v - w, v, w])

    def __repr__(self):
        return 'Element2D({}, {})'.format(self.level, self.vertices)

    def _create_new_vertex(self, nbr=None):
        """ Creates a new vertex necessary for bisection.

        Args:
            nbr : If self.refinement_edge() is not on the boundary, this
                   should be the element on the other side. 
        """
        assert self.is_leaf()
        vertex_parents = [self.newest_vertex()]
        if nbr:
            assert nbr.edge(0) == self.reversed_edge(0)
            vertex_parents.append(nbr.newest_vertex())

        godparents = self.vertices[1:]
        new_vertex = Vertex(vertex_parents[0].level + 1,
                            (godparents[0].x + godparents[1].x) / 2,
                            (godparents[0].y + godparents[1].y) / 2,
                            on_domain_boundary=nbr is None,
                            parents=vertex_parents)
        for vertex_parent in vertex_parents:
            vertex_parent.children.append(new_vertex)
        return new_vertex

    def _bisect(self, new_vertex=None):
        """ Bisects this element using Newest Vertex Bisection.

        Arguments:
            new_vertex: Reference to the new vertex. If none it is created..
        """
        assert self.is_leaf()
        if new_vertex is None:
            new_vertex = self._create_new_vertex()

        child1 = Element2D(self.level + 1,
                           [new_vertex, self.vertices[0], self.vertices[1]],
                           parent=self)
        child2 = Element2D(self.level + 1,
                           [new_vertex, self.vertices[2], self.vertices[0]],
                           parent=self)

        # NB: the neighbours along the newly created edges are not set.
        child1.neighbours = [self.neighbours[2], None, child2]
        child2.neighbours = [self.neighbours[1], child1, None]
        self.children = [child1, child2]
        new_vertex.patch.extend(self.children)

        # Also update the neighbours of *the neighbours* of the children;
        # it is currently set to `elem` but should be set to the child itself.
        for child in self.children:
            if child.neighbours[0] is not None:
                nbr_index = child.neighbours[0].neighbours.index(self)
                assert nbr_index is not None
                child.neighbours[0].neighbours[nbr_index] = child
                assert child in child.neighbours[0].neighbours

        return child1, child2

    def _bisect_with_nbr(self):
        """ Bisects this element and its neighbour -- ensures conformity. """
        nbr = self.neighbours[0]
        assert self.edge(0) == nbr.reversed_edge(0)

        new_vertex = self._create_new_vertex(nbr)
        child11, child12 = self._bisect(new_vertex)
        child21, child22 = nbr._bisect(new_vertex)

        # Set the neighbour info along shared edges of the children.
        child11.neighbours[1] = child22
        child12.neighbours[2] = child21
        child21.neighbours[1] = child12
        child22.neighbours[2] = child11


class InitialTriangulation:
    def __init__(self, vertices, elements):
        """ Instantiates the triangulation given the (vertices, elements).

        Arguments:
            vertices: Vx2 matrix of floats: the coordinates of the vertices.
            elements: Tx3 matrix of integers: the indices inside the vertices array.
        """
        self.vertex_roots = [
            Vertex(0, *vert, on_domain_boundary=False)
            for i, vert in enumerate(vertices)
        ]
        self.element_roots = [
            Element2D(0, [self.vertex_roots[idx] for idx in elem])
            for (i, elem) in enumerate(elements)
        ]
        self.elem_meta_root = MetaRoot(self.element_roots)
        self.vertex_meta_root = MetaRoot(self.vertex_roots)

        for idx, elem in enumerate(self.element_roots):
            # Set area.
            elem.area = self._compute_area(elem)
            # Set patch information.
            for v in elem.vertices:
                v.patch.append(elem)

            # Set initial neighbour information. Expensive.
            for nbr in self.element_roots[idx:]:
                for (i, j) in itertools.product(range(3), range(3)):
                    if elem.edge(i) == nbr.reversed_edge(j):
                        elem.neighbours[i] = nbr
                        nbr.neighbours[j] = elem
                        break

            # Set initial boundary information of vertices.
            for i, nbr in enumerate(elem.neighbours):
                if not nbr:
                    # No neighbour along edge `i` => both vertices on boundary.
                    for v in elem.edge(i):
                        v.on_domain_boundary = True

    @staticmethod
    def unit_square():
        """ Returns a (coarse) triangulation of the unit square. """
        vertices = [[0, 0], [1, 1], [1, 0], [0, 1]]
        elements = [[0, 2, 3], [1, 3, 2]]
        return InitialTriangulation(vertices, elements)

    def _compute_area(self, elem):
        """ Computes the area of the element spanned by `vertex_ids`. """
        v1 = elem.vertices[0].as_array()
        v2 = elem.vertices[1].as_array()
        v3 = elem.vertices[2].as_array()
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))


def to_matplotlib_triangulation(elem_meta_root, vertex_meta_root):
    """ The current triangulation as matplotlib-compatible object. """
    import matplotlib.tri as mpltri
    elements = [e for e in elem_meta_root.bfs() if e.is_leaf()]
    vertices = vertex_meta_root.bfs()
    if isinstance(vertices[0], NodeView):
        vertices = [v.node for v in vertices]
        elements = [e.node for e in elements]
    vertex_to_index = {}
    for idx, vertex in enumerate(vertices):
        vertex_to_index[vertex] = idx
    return mpltri.Triangulation([v.x for v in vertices],
                                [v.y for v in vertices],
                                [[vertex_to_index[v] for v in t.vertices]
                                 for t in elements])
