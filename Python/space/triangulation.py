import itertools

import matplotlib.pyplot as plt
import matplotlib.tri as mpltri
import numpy as np


class Vertex:
    """ A vertex in a locally refined triangulation. """

    def __init__(self, x, y, on_domain_boundary):
        self.x = x
        self.y = y
        self.on_domain_boundary = on_domain_boundary

    def as_array(self):
        return np.array([self.x, self.y], dtype=float)


class Triangle:
    """ A triangle as part of a locally refined triangulation. """

    def __init__(self, idx, vertex_ids, parent=None):
        """ Instantiates the triangle object.

        Arguments:
            idx: our index in the `triangulation.tris` array.
            vertex_ids: vertex id's in the `triangulation.verts` array.
            parent: reference to the parent of this triangle.
            """
        self.idx = idx
        self.vertex_ids = vertex_ids
        self.parent = parent
        self.children = []  # References to the children of this triangle.
        # Indices of my neighbours, ordered by the edge opposite vertex i.
        self.neighbours = [None, None, None]

        if self.parent:
            self.area = self.parent.area / 2

    def newest_vertex_id(self):
        """ Returns the newest vertex, i.e., vertex 0. """
        return self.vertex_ids[0]

    def edge(self, i):
        assert 0 <= i <= 2
        return [self.vertex_ids[(i + 1) % 3], self.vertex_ids[(i + 2) % 3]]

    def reversed_edge(self, i):
        assert 0 <= i <= 2
        return [self.vertex_ids[(i + 2) % 3], self.vertex_ids[(i + 1) % 3]]

    def is_leaf(self):
        return not len(self.children)


class Triangulation:
    def __init__(self, verts, tris):
        """ Instantiates the triangulation given the (vertices, triangles).

        Arguments:
            verts: Vx2 matrix of floats: the coordinates of the vertices.
            tris: Tx3 matrix of integers: the indices inside the verts array.
        """
        self.verts = [
            Vertex(*vert, on_domain_boundary=False) for vert in verts
        ]
        self.tris = [Triangle(i, tri) for (i, tri) in enumerate(tris)]
        self.root_ids = set([tri.idx for tri in self.tris])
        self.history = []

        for tri in self.tris:
            tri.area = self._compute_area(tri)

        # Set initial neighbour information. Expensive.
        for tri in self.tris:
            for nbr in self.tris[tri.idx:]:  # Small optimization.
                for (i, j) in itertools.product(range(3), range(3)):
                    if tri.edge(i) == nbr.reversed_edge(j):
                        tri.neighbours[i] = nbr
                        nbr.neighbours[j] = tri
                        break

        # Set initial boundary information of vertices.
        for tri in self.tris:
            for i, nbr in enumerate(tri.neighbours):
                if not nbr:
                    # No neighbour along edge `i` => both vertices on boundary.
                    for v in tri.edge(i):
                        self.verts[v].on_domain_boundary = True

    @staticmethod
    def unit_square():
        """ Returns a (coarse) triangulation of the unit square. """
        verts = [[0, 0], [1, 1], [1, 0], [0, 1]]
        tris = [[0, 2, 3], [1, 3, 2]]
        return Triangulation(verts, tris)

    def _bisect(self, tri, new_vertex_id=False):
        """ Bisects a given triangle using Newest Vertex Bisection.

        Arguments:
            tri: reference to a Triangle object.
            new_vertex_id: index of the new vertex in the `self.verts` array. If
                not passed, creates a new vertex.
        """
        vert_ids = tri.vertex_ids
        if not new_vertex_id:
            # Create the new vertex object.
            parents = [self.verts[v] for v in vert_ids[1:]]
            on_domain_boundary = all([p.on_domain_boundary for p in parents])
            new_vertex = Vertex((parents[0].x + parents[1].x) / 2,
                                (parents[0].y + parents[1].y) / 2,
                                on_domain_boundary)
            self.verts.append(new_vertex)
            new_vertex_id = len(self.verts) - 1
            self.history.append((new_vertex_id, tri.idx))
        child1_id = len(self.tris)
        child2_id = len(self.tris) + 1
        child1 = Triangle(
            child1_id, [new_vertex_id, vert_ids[0], vert_ids[1]], parent=tri)
        child2 = Triangle(
            child2_id, [new_vertex_id, vert_ids[2], vert_ids[0]], parent=tri)

        self.tris.extend([child1, child2])
        # NB: the neighbours along the newly created edges are not set.
        child1.neighbours = [tri.neighbours[2], None, child2]
        child2.neighbours = [tri.neighbours[1], child1, None]
        tri.children = [child1, child2]

        # Also update the neighbours of *the neighbours* of the children;
        # it is currently set to `tri` but should be set to the child itself.
        for child in tri.children:
            if child.neighbours[0] is not None:
                nbr_index = child.neighbours[0].neighbours.index(tri)
                assert nbr_index is not None
                child.neighbours[0].neighbours[nbr_index] = child
                assert child in child.neighbours[0].neighbours

        return child1, child2

    def _bisect_pair(self, tri1, tri2):
        """ Bisects a pair of triangles. Sets the childrens neighbours. """
        assert tri1.edge(0) == tri2.reversed_edge(0)
        child11, child12 = self._bisect(tri1)  # child11s newest vertex is new.
        child21, child22 = self._bisect(
            tri2, new_vertex_id=child11.newest_vertex_id())
        # Set the neighbour info along shared edges of the children.
        child11.neighbours[1] = child22
        child12.neighbours[2] = child21
        child21.neighbours[1] = child12
        child22.neighbours[2] = child11
        self.verts[child11.newest_vertex_id()].on_domain_boundary = False

    def refine(self, tri):
        """ Refines the trianglulation so that triangle `tri` is bisected.

        If this triangle was already bisected, then this function has no effect.
        """
        if not tri.is_leaf():
            return
        nbr = tri.neighbours[0]
        if not nbr:  # Refinement edge of `tri` is on domain boundary.
            self._bisect(tri)
            return
        if nbr.edge(0) == tri.reversed_edge(0):  # Shared refinement edge.
            self._bisect_pair(tri, nbr)
        else:
            self.refine(nbr)
            if nbr.children[0].edge(0) == tri.reversed_edge(0):
                self._bisect_pair(nbr.children[0], tri)
            else:
                self._bisect_pair(nbr.children[1], tri)

    def refine_uniform(self):
        """ Performs a uniform refinement on the triangulation. """
        leaves = [tri for tri in self.tris if tri.is_leaf()]
        for leaf in leaves:
            self.refine(leaf)

    def _compute_area(self, tri):
        """ Computes the area of the triangle spanned by `vertex_ids`. """
        v1 = self.verts[tri.vertex_ids[0]].as_array()
        v2 = self.verts[tri.vertex_ids[1]].as_array()
        v3 = self.verts[tri.vertex_ids[2]].as_array()
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    def as_matplotlib_triangulation(self):
        """ The current triangulation as matplotlib-compatible object. """
        print([v.x for v in self.verts], [v.y for v in self.verts],
              [t.vertex_ids for t in self.tris if t.is_leaf()])
        return mpltri.Triangulation(
            [v.x for v in self.verts], [v.y for v in self.verts],
            [t.vertex_ids for t in self.tris if t.is_leaf()])


def plot_hatfn():
    triangulation = Triangulation.unit_square()
    triangulation.refine(triangulation.tris[0])
    triangulation.refine(triangulation.tris[4])
    triangulation.refine(triangulation.tris[7])
    triangulation.refine(triangulation.tris[2])

    print(triangulation.history)
    I = np.eye(len(triangulation.verts))
    for i in range(len(triangulation.verts)):
        fig = plt.figure()
        fig.suptitle("Hoedfuncties bij vertex %d" % i)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title("Nodale basis")
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title("Hierarchische basis")
        ax1.plot_trisurf(
            triangulation.as_matplotlib_triangulation(), Z=I[:, i])
        w = triangulation.apply_T(I[:, i])
        ax2.plot_trisurf(triangulation.as_matplotlib_triangulation(), Z=w)
        plt.show()


if __name__ == "__main__":
    test_galerkin(plot=True)
