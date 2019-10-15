import random

import numpy as np
import quadpy

from ..datastructures.tree_vector import TreeVector
from ..datastructures.tree_view import TreeView
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
            vec_in = TreeVector(T.vertex_meta_root)
            vec_in.uniform_refine(ml)
            assert len(vec_in.bfs()) < len(T.vertex_meta_root.bfs())
            for vertex in vec_in.bfs():
                vertex.value = random.random()

            vec_out = TreeVector(T.vertex_meta_root)
            vec_out.uniform_refine(max_level=ml)
            applicator.apply(vec_in, vec_out)

            # Compare.
            T_view = TriangulationView(vec_in)
            assert vec_in.bfs() == T_view.vertices
            vec_out_np = op.apply(vec_in.to_array())
            assert np.allclose(vec_out.to_array(), vec_out_np)


def test_mass_quad_non_symmetric():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)

    # Create a vector with only vertices lying below the diagonal upto lvl 4
    Lambda_below = TreeView(T.vertex_meta_root)
    Lambda_below.deep_refine(call_filter=lambda vertex: vertex.level <= 4 and
                             vertex.x + vertex.y <= 1)

    # Create an out vector with only vertices lying above the diagonal.
    Lambda_above = TreeVector(T.vertex_meta_root)
    Lambda_above.deep_refine(
        call_filter=lambda vertex: vertex.x + vertex.y >= 1)

    assert len(Lambda_below.bfs()) < len(Lambda_above.bfs())

    for op, deriv in [(MassOperator(), False), (StiffnessOperator(), True)]:
        applicator = Applicator(op)

        for Lambda_in, Lambda_out in [(Lambda_below, Lambda_below),
                                      (Lambda_below, Lambda_above),
                                      (Lambda_above, Lambda_below)]:

            # Matrix
            mat = applicator.to_matrix(Lambda_in, Lambda_out)

            # Compare with quadrature
            quad_scheme = quadpy.triangle.newton_cotes_open(0 if deriv else 2)
            for i, v_psi in enumerate(Lambda_in.bfs()):
                psi = HierarchicalBasisFunction(v_psi.node)
                for j, v_phi in enumerate(Lambda_out.bfs()):
                    phi = HierarchicalBasisFunction(v_phi.node)
                    support = psi.support if psi.level > phi.level else phi.support
                    real_ip = 0
                    for elem in support:
                        triangle = np.array(
                            [elem.vertices[i].as_array() for i in range(3)])
                        func = lambda x: (psi.eval(x, deriv) * phi.eval(
                            x, deriv)).sum(axis=0)
                        real_ip += quad_scheme.integrate(func, triangle)

                    print(psi, phi, deriv)
                    assert np.allclose(real_ip, mat[j, i])
