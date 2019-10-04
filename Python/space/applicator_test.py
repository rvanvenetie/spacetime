import random

import numpy as np
import quadpy

from ..datastructures.tree_vector import MetaRootVector, NodeVector
from ..datastructures.tree_view import MetaRootView
from .applicator import Applicator
from .basis import HierarchicalBasisFunction
from .operators import MassOperator, StiffnessOperator
from .triangulation import InitialTriangulation
from .triangulation_view import TriangulationView


def applicator_to_matrix(applicator, Lambda_in, Lambda_out):
    nodes_in = Lambda_in.bfs()
    nodes_out = Lambda_out.bfs()

    n, m = len(nodes_out), len(nodes_in)
    result = np.zeros((n, m))
    for i, psi in enumerate(nodes_in):
        # Create vector with a 1 for psi
        vec_in = Lambda_in.deep_copy(nv_cls=NodeVector, mv_cls=MetaRootVector)
        for n in vec_in.bfs():
            if n.node == psi.node:
                n.value = 1
                break
        assert sum(n.value for n in vec_in.bfs()) == 1

        vec_out = Lambda_out.deep_copy(nv_cls=NodeVector,
                                       mv_cls=MetaRootVector)
        assert sum(n.value for n in vec_out.bfs()) == 0
        applicator.apply(vec_in, vec_out)
        for j, phi in enumerate(vec_out.bfs()):
            result[j, i] = phi.value
    return result


def test_operators():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(8)
    for op in [MassOperator(), StiffnessOperator()]:
        applicator = Applicator(op)
        for ml in range(5):
            # Create some (sub)vector on the vertices.
            vec_in = MetaRootVector(T.vertex_meta_root)
            vec_in.uniform_refine(ml)
            assert len(vec_in.bfs()) < len(T.vertex_meta_root.bfs())
            for vertex in vec_in.bfs():
                vertex.value = random.random()

            vec_out = MetaRootVector(T.vertex_meta_root)
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
    Lambda_below = MetaRootView(T.vertex_meta_root)
    Lambda_below.deep_refine(call_filter=lambda vertex: vertex.level <= 4 and
                             vertex.x + vertex.y <= 1)

    # Create an out vector with only vertices lying above the diagonal.
    Lambda_above = MetaRootVector(T.vertex_meta_root)
    Lambda_above.deep_refine(
        call_filter=lambda vertex: vertex.x + vertex.y >= 1)

    assert len(Lambda_below.bfs()) < len(Lambda_above.bfs())

    for op, deriv in [(MassOperator(), False), (StiffnessOperator(), True)]:
        applicator = Applicator(op)

        for Lambda_in, Lambda_out in [(Lambda_below, Lambda_below),
                                      (Lambda_below, Lambda_above),
                                      (Lambda_above, Lambda_below)]:

            # Matrix
            mat = applicator_to_matrix(applicator, Lambda_in, Lambda_out)

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

                        def func(x):
                            return np.array([
                                np.dot(psi.eval(x[:, i], deriv),
                                       phi.eval(x[:, i], deriv))
                                for i in range(x.shape[1])
                            ])

                        real_ip += quad_scheme.integrate(func, triangle)
                    assert np.allclose(real_ip, mat[j, i])
