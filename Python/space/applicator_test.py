import random

import numpy as np

from ..datastructures.tree_vector import TreeVector
from .applicator import Applicator
from .basis import HierarchicalBasisFunction
from .operators import MassOperator, StiffnessOperator
from .triangulation import InitialTriangulation
from .triangulation_view import TriangulationView


def test_operators():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(8)
    for op in [MassOperator(), StiffnessOperator()]:
        applicator = Applicator(op)
        for ml in range(5):
            # Create some (sub)vector on the vertices.
            vec_in = TreeVector.from_metaroot(T.vertex_meta_root)
            vec_in.uniform_refine(ml)
            assert len(vec_in.bfs()) < len(T.vertex_meta_root.bfs())
            for vertex in vec_in.bfs():
                vertex.value = random.random()

            vec_out = TreeVector.from_metaroot(T.vertex_meta_root)
            vec_out.uniform_refine(max_level=ml)
            applicator.apply(vec_in, vec_out)

            # Compare.
            T_view = TriangulationView(vec_in)
            assert [nv.node for nv in vec_in.bfs()] == T_view.vertices
            vec_out_np = op.apply(vec_in.to_array())
            assert np.allclose(vec_out.to_array(), vec_out_np)


def test_mass_quad_non_symmetric():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)

    # Create a vector with only vertices lying below the diagonal upto lvl 4
    Lambda_below = HierarchicalBasisFunction.from_triangulation(T)
    Lambda_below.deep_refine(call_filter=lambda vertex: vertex.level <= 4 and
                             vertex.x + vertex.y <= 1)

    # Create an out vector with only vertices lying above the diagonal.
    Lambda_above = HierarchicalBasisFunction.from_triangulation(T)
    Lambda_above.deep_refine(
        call_filter=lambda vertex: vertex.x + vertex.y >= 1)

    assert len(Lambda_below.bfs()) < len(Lambda_above.bfs())

    for op, deriv in [(MassOperator(dirichlet_boundary=False), False),
                      (StiffnessOperator(dirichlet_boundary=False), True)]:
        applicator = Applicator(op)

        for Lambda_in, Lambda_out in [(Lambda_below, Lambda_below),
                                      (Lambda_below, Lambda_above),
                                      (Lambda_above, Lambda_below)]:

            # Matrix
            mat = applicator.to_matrix(Lambda_in, Lambda_out)

            # Compare with quadrature
            for i, psi in enumerate(Lambda_in.bfs()):
                for j, phi in enumerate(Lambda_out.bfs()):
                    f, g = (psi, phi) if psi.level > phi.level else (phi, psi)
                    real_ip = f.inner_quad(lambda xy: g.eval(xy, deriv=deriv),
                                           g_order=1,
                                           deriv=deriv)
                    assert np.allclose(real_ip, mat[j, i])


def test_dirichlet_boundary():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)

    # Create a vector with only vertices lying below the diagonal upto lvl 4
    Lambda_below = HierarchicalBasisFunction.from_triangulation(T)
    Lambda_below.deep_refine(call_filter=lambda vertex: vertex.level <= 4 and
                             vertex.x + vertex.y <= 1)

    # Create an out vector with only vertices lying above the diagonal.
    Lambda_above = HierarchicalBasisFunction.from_triangulation(T)
    Lambda_above.deep_refine(
        call_filter=lambda vertex: vertex.x + vertex.y >= 1)

    for op in [
            MassOperator(dirichlet_boundary=True),
            StiffnessOperator(dirichlet_boundary=True)
    ]:
        applicator = Applicator(op)

        for Lambda_in, Lambda_out in [(Lambda_below, Lambda_below),
                                      (Lambda_below, Lambda_above),
                                      (Lambda_above, Lambda_below)]:
            # Matrix
            mat = applicator.to_matrix(Lambda_in, Lambda_out)

            # Compare with quadrature
            for i, psi in enumerate(Lambda_in.bfs()):
                for j, phi in enumerate(Lambda_out.bfs()):
                    if psi.node.on_domain_boundary or phi.node.on_domain_boundary:
                        assert mat[j, i] == 0
