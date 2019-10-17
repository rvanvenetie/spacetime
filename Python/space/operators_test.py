import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import cg

from ..datastructures.tree_view import NodeView, TreeView
from .operators import MassOperator, Operator, StiffnessOperator
from .triangulation import InitialTriangulation, to_matplotlib_triangulation
from .triangulation_view import TriangulationView


def test_transformation():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.bfs()[0].refine()
    T.elem_meta_root.bfs()[4].refine()
    T.elem_meta_root.bfs()[7].refine()
    vertices = T.vertex_meta_root.bfs()
    elements = T.elem_meta_root.bfs()

    assert len(vertices) == 8
    assert len([elem for elem in elements if elem.is_leaf()]) == 8

    # Create a view of the vertices.
    vertex_view = TreeView.from_metaroot(T.vertex_meta_root)
    vertex_view.deep_refine()
    T_view = TriangulationView(vertex_view)
    operators = Operator(T_view)
    v = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    w = np.array([0, 0, 0, 1, 0.5, 0.5, 0.5, 0.75], dtype=float)
    w2 = operators.apply_T(v)
    assert norm(w - w2) < 1e-10

    v = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    w = np.array([0, 0, 0.25, 0.75, 0.5, 0, 0, 1], dtype=float)
    w2 = operators.apply_T_transpose(v)
    assert norm(w - w2) < 1e-10

    for _ in range(10):
        v = np.random.rand(8)
        z = np.random.rand(8)
        # Test that <applyT(v), z> = <v, applyT_transpose(z)>.
        assert norm(
            np.inner(v, operators.apply_T_transpose(z)) -
            np.inner(operators.apply_T(v), z)) < 1e-10
        assert norm(v -
                    operators.apply_T(operators.apply_T_inverse(v))) < 1e-10
        assert norm(v -
                    operators.apply_T_inverse(operators.apply_T(v))) < 1e-10

        # Test that T is a linear operator.
        alpha = np.random.rand()
        assert norm(
            operators.apply_T(v + alpha * z) -
            (operators.apply_T(v) + alpha * operators.apply_T(z))) < 1e-10


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
    vertices = [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    elements = [[0, 2, 3], [1, 3, 2]]
    T = InitialTriangulation(vertices, elements)
    vertex_view = TreeView.from_metaroot(T.vertex_meta_root)
    vertex_view.deep_refine()
    # Create a view of the vertices.
    T_view = TriangulationView(vertex_view)
    ones = np.ones(len(T_view.vertices), dtype=float)
    mass_op = MassOperator(T_view)
    rhs = mass_op.apply_T_transpose(mass_op.apply_SS(ones))

    for i in range(9):
        T.elem_meta_root.uniform_refine(i)
        vertex_view.deep_refine()
        T_view = TriangulationView(vertex_view)
        mass_op.triang = T_view
        ones = np.ones(len(T_view.vertices), dtype=float)
        new_rhs = mass_op.apply_T_transpose(mass_op.apply_SS(ones))
        # Test that the first V elements of the right-hand side coincide -- we
        # have a hierarchic basis after all.
        assert norm(rhs - rhs[:rhs.shape[0]]) < 1e-10
        rhs = new_rhs

    rhs = mass_op.apply_boundary_restriction(rhs)
    stiff_op = StiffnessOperator(T_view)
    stiff_op_scipy = stiff_op.as_boundary_restricted_linear_operator()
    sol_HB, _ = cg(stiff_op_scipy, rhs, atol=0, tol=1e-8)
    sol_SS = stiff_op.apply_T(sol_HB)

    assert np.abs(sol_SS[4] - 0.2946854131260553) < 1e-3  # solution in (0, 0).
    for i in [9, 10, 11, 12]:
        assert np.abs(sol_SS[i] - 0.181145) < 1e-3  # solution in (0.5, 0.5).

    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(to_matplotlib_triangulation(T.elem_meta_root,
                                                    T.vertex_meta_root),
                        Z=sol_SS)
        plt.show()


if __name__ == "__main__":
    test_galerkin(plot=True)
