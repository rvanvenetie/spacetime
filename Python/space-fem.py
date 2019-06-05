import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mpltri
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
import time
from scipy.sparse.linalg import LinearOperator, cg


class Vertex(object):
    """ A vertex in a locally refined triangulation. """

    def __init__(self, x, y, on_domain_boundary):
        self.x = x
        self.y = y
        self.on_domain_boundary = on_domain_boundary

    def as_array(self):
        return np.array([self.x, self.y], dtype=float)


class Triangle(object):
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


class Triangulation(object):
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
        child1 = Triangle(child1_id, [new_vertex_id, vert_ids[0], vert_ids[1]],
                          parent=tri)
        child2 = Triangle(child2_id, [new_vertex_id, vert_ids[2], vert_ids[0]],
                          parent=tri)

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

    def apply_T(self, v):
        """
        Applies the hierarchical-to-single-scale transformation to a vector `v`.

        Arguments:
            v: a `np.array` of length len(self.verts).

        Returns:
            w: a `np.array` of length len(self.verts).
        """
        w = np.copy(v)
        for (vi, Ti) in self.history:
            godfather_vertices = self.tris[Ti].edge(0)
            for gf in godfather_vertices:
                w[vi] = w[vi] + 0.5 * w[gf]
        return w

    def apply_T_transpose(self, v):
        """
        Applies the transposed hierarchical-to-single-scale transformation to `v`.

        Arguments:
            v: a `np.array` of length len(self.verts).

        Returns:
            w: a `np.array` of length len(self.verts).
        """
        w = np.copy(v)
        for (vi, Ti) in reversed(self.history):
            godfather_vertices = self.tris[Ti].edge(0)
            for gf in godfather_vertices:
                w[gf] = w[gf] + 0.5 * w[vi]
        return w

    def _compute_area(self, tri):
        """ Computes the area of the triangle spanned by `vertex_ids`. """
        v1 = self.verts[tri.vertex_ids[0]].as_array()
        v2 = self.verts[tri.vertex_ids[1]].as_array()
        v3 = self.verts[tri.vertex_ids[2]].as_array()
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    def apply_SS_mass(self, v):
        """ Applies the single-scale mass matrix. """
        element_mass = 1.0 / 12.0 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        w = np.zeros(v.shape)
        for tri in self.tris:
            if not tri.is_leaf():
                continue

            Vids = tri.vertex_ids
            for (i, j) in itertools.product(range(3), range(3)):
                w[Vids[j]] += element_mass[i, j] * tri.area * v[Vids[i]]
        return w

    def apply_SS_stiffness(self, v):
        """ Applies the single-scale stiffness matrix. """
        w = np.zeros(v.shape)
        for tri in self.tris:
            if not tri.is_leaf():
                continue
            Vids = tri.vertex_ids
            V = [self.verts[idx] for idx in Vids]
            D = np.array([[V[2].x - V[1].x, V[0].x - V[2].x, V[1].x - V[0].x],
                          [V[2].y - V[1].y, V[0].y - V[2].y, V[1].y - V[0].y]],
                         dtype=float)
            element_stiff = (D.T @ D) / (4 * tri.area)
            for (i, j) in itertools.product(range(3), range(3)):
                w[Vids[j]] += element_stiff[i, j] * v[Vids[i]]
        return w

    def apply_HB_mass(self, v):
        """ Applies the hierarchical mass matrix. """
        w = self.apply_T(v)
        x = self.apply_SS_mass(w)
        return self.apply_T_transpose(x)

    def apply_HB_stiffness(self, v):
        """ Applies the hierarchical stiffness matrix. """
        w = self.apply_T(v)
        x = self.apply_SS_stiffness(w)
        return self.apply_T_transpose(x)

    def apply_boundary_restriction(self, v):
        """ Sets all boundary vertices to zero. """
        w = np.zeros(v.shape)
        for i in range(v.shape[0]):
            w[i] = v[i] if not self.verts[i].on_domain_boundary else 0.0
        return w

    def as_linear_operator(self, method):
        """ Recasts the application of a method as a scipy LinearOperator. """
        return LinearOperator(dtype=float,
                              shape=(len(self.verts), len(self.verts)),
                              matvec=lambda x: method(x))

    def as_boundary_restricted_linear_operator(self, method):
        """ Recasts the application of a method as a scipy LinearOperator. """
        return LinearOperator(
            dtype=float,
            shape=(len(self.verts), len(self.verts)),
            matvec=lambda x: self.apply_boundary_restriction(method(x)))

    def as_matplotlib_triangulation(self):
        """ The current triangulation as matplotlib-compatible object. """
        print([v.x for v in self.verts], [v.y for v in self.verts],
              [t.vertex_ids for t in self.tris if t.is_leaf()])
        return mpltri.Triangulation(
            [v.x for v in self.verts], [v.y for v in self.verts],
            [t.vertex_ids for t in self.tris if t.is_leaf()])


def test_transformation():
    verts = [[0, 0], [1, 1], [1, 0], [0, 1]]
    tris = [[0, 2, 3], [1, 3, 2]]
    triangulation = Triangulation(verts, tris)
    triangulation.refine(triangulation.tris[0])
    triangulation.refine(triangulation.tris[4])
    triangulation.refine(triangulation.tris[7])

    assert len(triangulation.verts) == 8
    assert len([tri for tri in triangulation.tris if tri.is_leaf()]) == 8
    assert len(triangulation.history) == 4

    v = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    w = np.array([0, 0, 0, 1, 0.5, 0.5, 0.5, 0.75], dtype=float)
    w2 = triangulation.apply_T(v)
    assert norm(w - w2) < 1e-10

    v = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    w = np.array([0, 0, 0.25, 0.75, 0.5, 0, 0, 1], dtype=float)
    w2 = triangulation.apply_T_transpose(v)
    assert norm(w - w2) < 1e-10

    for _ in range(10):
        v = np.random.rand(8)
        z = np.random.rand(8)
        # Test that <applyT(v), z> = <v, applyT_transpose(z)>.
        assert norm(
            np.inner(v, triangulation.apply_T_transpose(z)) -
            np.inner(triangulation.apply_T(v), z)) < 1e-10

        # Test that T is a linear operator.
        alpha = np.random.rand()
        assert norm(
            triangulation.apply_T(v + alpha * z) -
            (triangulation.apply_T(v) +
             alpha * triangulation.apply_T(z))) < 1e-10


def test_on_domain_bdr():
    verts = [[0, 0], [1, 1], [1, 0], [0, 1]]
    tris = [[0, 2, 3], [1, 3, 2]]
    triangulation = Triangulation(verts, tris)
    assert all([v.on_domain_boundary for v in triangulation.verts])
    triangulation.refine(triangulation.tris[0])
    for _ in range(100):
        triangulation.refine(triangulation.tris[np.random.randint(
            len(triangulation.verts))])
    for vert in triangulation.verts:
        assert vert.on_domain_boundary == (vert.x == 0 or vert.x == 1
                                           or vert.y == 0 or vert.y == 1)


def test_galerkin(plot=False):
    """ Tests -Laplace u = 1 on [-1,1]^2 with zero boundary conditions.
    
    From http://people.inf.ethz.ch/arbenz/FEM17/pdfs/0-19-852868-X.pdf,
    we find the analytical solution. We use Mathematica to compute a few
    values with adequate precision, through
     | K_max := 10001
     | u[x_, y_] := (1 - x^2)/2 - 16/Pi^3 *
     |     Sum[Sin[k Pi (1 + x)/2]/(k^3 Sinh[k Pi]) * (Sinh[k Pi (1 + y)/2] +
     |         Sinh[k Pi (1 - y)/2]), {k, 1, K_max, 2}]
    and verify that our solution comes fairly close to this solution in a
    couply of points.
    """
    verts = [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    tris = [[0, 2, 3], [1, 3, 2]]
    triangulation = Triangulation(verts, tris)
    ones = np.ones(len(triangulation.verts), dtype=float)
    rhs = triangulation.apply_T_transpose(triangulation.apply_SS_mass(ones))

    for _ in range(9):
        triangulation.refine_uniform()
        ones = np.ones(len(triangulation.verts), dtype=float)
        new_rhs = triangulation.apply_T_transpose(
            triangulation.apply_SS_mass(ones))
        # Test that the first V elements of the right-hand side coincide -- we
        # have a hierarchic basis after all.
        assert norm(rhs - rhs[:rhs.shape[0]]) < 1e-10
        rhs = new_rhs

    rhs = triangulation.apply_boundary_restriction(rhs)
    stiff = triangulation.as_boundary_restricted_linear_operator(
        triangulation.apply_HB_stiffness)
    sol_HB, _ = cg(stiff, rhs, atol=0, tol=1e-8)
    sol_SS = triangulation.apply_T(sol_HB)

    assert np.abs(sol_SS[4] - 0.2946854131260553) < 1e-3  # solution in (0, 0).
    for i in [9, 10, 11, 12]:
        assert np.abs(sol_SS[i] - 0.181145) < 1e-3  # solution in (0.5, 0.5).

    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(triangulation.as_matplotlib_triangulation(), Z=sol_SS)
        plt.show()


def plot_hatfn():
    verts = [[0, 0], [1, 1], [1, 0], [0, 1]]
    tris = [[0, 2, 3], [1, 3, 2]]
    triangulation = Triangulation(verts, tris)
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
        ax1.plot_trisurf(triangulation.as_matplotlib_triangulation(),
                         Z=I[:, i])
        w = triangulation.apply_T(I[:, i])
        ax2.plot_trisurf(triangulation.as_matplotlib_triangulation(), Z=w)
        plt.show()


if __name__ == "__main__":
    test_galerkin(plot=True)
