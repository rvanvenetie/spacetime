import itertools

import matplotlib.pyplot as plt
import matplotlib.tri as mpltri
import numpy as np


class Vertex:
    """ A vertex in a locally refined triangulation. """
    def __init__(self, idx, x, y, on_domain_boundary):
        self.idx = idx
        self.x = x
        self.y = y
        self.on_domain_boundary = on_domain_boundary

    def as_array(self):
        return np.array([self.x, self.y], dtype=float)


class Element:
    """ A element as part of a locally refined triangulation. """
    def __init__(self, idx, vertices, parent=None):
        """ Instantiates the element object.

        Arguments:
            idx: our index in the `triangulation.elements` array.
            vertices: array of three Vertex references. 
            parent: reference to the parent of this element.
            """
        self.idx = idx
        self.vertices = vertices
        self.parent = parent
        self.children = []  # References to the children of this element.
        # Indices of my neighbours, ordered by the edge opposite vertex i.
        self.neighbours = [None, None, None]

        if self.parent:
            self.area = self.parent.area / 2

    def newest_vertex_id(self):
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


class Triangulation:
    def __init__(self, vertices, elements):
        """ Instantiates the triangulation given the (vertices, elements).

        Arguments:
            vertices: Vx2 matrix of floats: the coordinates of the vertices.
            elements: Tx3 matrix of integers: the indices inside the vertices array.
        """
        self.vertices = [
            Vertex(idx, *vert, on_domain_boundary=False)
            for idx, vert in enumerate(vertices)
        ]
        self.elements = [
            Element(i, [self.vertices[idx] for idx in elem])
            for (i, elem) in enumerate(elements)
        ]
        self.root_ids = set([elem.idx for elem in self.elements])
        self.history = []

        for elem in self.elements:
            elem.area = self._compute_area(elem)

        # Set initial neighbour information. Expensive.
        for elem in self.elements:
            for nbr in self.elements[elem.idx:]:  # Small optimization.
                for (i, j) in itertools.product(range(3), range(3)):
                    if elem.edge(i) == nbr.reversed_edge(j):
                        elem.neighbours[i] = nbr
                        nbr.neighbours[j] = elem
                        break

        # Set initial boundary information of vertices.
        for elem in self.elements:
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
        return Triangulation(vertices, elements)

    def _bisect(self, elem, new_vertex=False):
        """ Bisects a given element using Newest Vertex Bisection.

        Arguments:
            elem: reference to a Element object.
            new_vertex_id: index of the new vertex in the `self.vertices` array. If
                not passed, creates a new vertex.
        """
        vertices = elem.vertices
        if not new_vertex:
            # Create the new vertex object.
            parents = vertices[1:]
            on_domain_boundary = all([p.on_domain_boundary for p in parents])
            new_vertex_id = len(self.vertices)
            new_vertex = Vertex(new_vertex_id,
                                (parents[0].x + parents[1].x) / 2,
                                (parents[0].y + parents[1].y) / 2,
                                on_domain_boundary)
            self.vertices.append(new_vertex)
            self.history.append((new_vertex_id, elem.idx))
        child1_id = len(self.elements)
        child2_id = len(self.elements) + 1
        child1 = Element(child1_id, [new_vertex, vertices[0], vertices[1]],
                         parent=elem)
        child2 = Element(child2_id, [new_vertex, vertices[2], vertices[0]],
                         parent=elem)

        self.elements.extend([child1, child2])
        # NB: the neighbours along the newly created edges are not set.
        child1.neighbours = [elem.neighbours[2], None, child2]
        child2.neighbours = [elem.neighbours[1], child1, None]
        elem.children = [child1, child2]

        # Also update the neighbours of *the neighbours* of the children;
        # it is currently set to `elem` but should be set to the child itself.
        for child in elem.children:
            if child.neighbours[0] is not None:
                nbr_index = child.neighbours[0].neighbours.index(elem)
                assert nbr_index is not None
                child.neighbours[0].neighbours[nbr_index] = child
                assert child in child.neighbours[0].neighbours

        return child1, child2

    def _bisect_pair(self, elem1, elem2):
        """ Bisects a pair of elemangles. Sets the childrens neighbours. """
        assert elem1.edge(0) == elem2.reversed_edge(0)
        child11, child12 = self._bisect(
            elem1)  # child11s newest vertex is new.
        child21, child22 = self._bisect(elem2,
                                        new_vertex=child11.newest_vertex_id())
        # Set the neighbour info along shared edges of the children.
        child11.neighbours[1] = child22
        child12.neighbours[2] = child21
        child21.neighbours[1] = child12
        child22.neighbours[2] = child11
        child11.newest_vertex_id().on_domain_boundary = False

    def refine(self, elem):
        """ Refines the trianglulation so that element `elem` is bisected.

        If this triangle was already bisected, then this function has no effect.
        """
        if not elem.is_leaf():
            return
        nbr = elem.neighbours[0]
        if not nbr:  # Refinement edge of `elem` is on domain boundary.
            self._bisect(elem)
            return
        if nbr.edge(0) == elem.reversed_edge(0):  # Shared refinement edge.
            self._bisect_pair(elem, nbr)
        else:
            self.refine(nbr)
            if nbr.children[0].edge(0) == elem.reversed_edge(0):
                self._bisect_pair(nbr.children[0], elem)
            else:
                self._bisect_pair(nbr.children[1], elem)

    def refine_uniform(self):
        """ Performs a uniform refinement on the triangulation. """
        leaves = [elem for elem in self.elements if elem.is_leaf()]
        for leaf in leaves:
            self.refine(leaf)

    def _compute_area(self, elem):
        """ Computes the area of the element spanned by `vertex_ids`. """
        v1 = elem.vertices[0].as_array()
        v2 = elem.vertices[1].as_array()
        v3 = elem.vertices[2].as_array()
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    def as_matplotlib_triangulation(self):
        """ The current triangulation as matplotlib-compatible object. """
        print([v.x for v in self.vertices], [v.y for v in self.vertices],
              [t.vertex_ids for t in self.elements if t.is_leaf()])
        return mpltri.Triangulation(
            [v.x for v in self.vertices], [v.y for v in self.vertices],
            [t.vertex_ids for t in self.elements if t.is_leaf()])


def plot_hatfn():
    triangulation = Triangulation.unit_square()
    triangulation.refine(triangulation.elements[0])
    triangulation.refine(triangulation.elements[4])
    triangulation.refine(triangulation.elements[7])
    triangulation.refine(triangulation.elements[2])

    print(triangulation.history)
    I = np.eye(len(triangulation.vertices))
    for i in range(len(triangulation.vertices)):
        fig = plt.figure()
        fig.suptitle("Hoedfuncties bij vertex %d" % i)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title("Nodale basis")
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title("Hierarchische basis")
        ax1.plot_elementsurf(triangulation.as_matplotlib_triangulation(),
                             Z=I[:, i])
        w = triangulation.apply_T(I[:, i])
        ax2.plot_elementsurf(triangulation.as_matplotlib_triangulation(), Z=w)
        plt.show()


if __name__ == "__main__":
    test_galerkin(plot=True)
