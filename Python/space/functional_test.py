from pytest import approx

from .basis import HierarchicalBasisFunction
from .functional import Functional
from .operators import QuadratureOperator
from .triangulation import InitialTriangulation


def test_functional_quadrature():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)

    Lambda_uniform = HierarchicalBasisFunction.from_triangulation(T)
    Lambda_uniform.deep_refine()

    # Create a vector with only vertices lying below the diagonal upto lvl 4
    Lambda_below = HierarchicalBasisFunction.from_triangulation(T)
    Lambda_below.deep_refine(call_filter=lambda vertex: vertex.level <= 4 and
                             vertex.x + vertex.y <= 1)

    # Create an out vector with only vertices lying above the diagonal.
    Lambda_above = HierarchicalBasisFunction.from_triangulation(T)
    Lambda_above.deep_refine(
        call_filter=lambda vertex: vertex.x + vertex.y >= 1)

    assert len(Lambda_below.bfs()) < len(Lambda_above.bfs()) < len(
        Lambda_uniform.bfs())

    for Lambda in [Lambda_uniform, Lambda_below, Lambda_above]:
        for g, g_order, deriv in [(lambda xy: (xy[0] - 1) * xy[0] +
                                   (xy[1] - 1) * xy[1], 4, False),
                                  (lambda xy: xy[0] * xy[1] -
                                   (xy[0] + xy[1]) / 2, 2, False),
                                  (lambda xy: xy * xy * xy, 3, True)]:
            operator = QuadratureOperator(g=g, g_order=g_order, deriv=deriv)
            functional = Functional(operator)

            inner_g_vec = functional.eval(Lambda)
            for psi in inner_g_vec.bfs():
                true_ip = HierarchicalBasisFunction(psi.node).inner_quad(
                    g=g, g_order=g_order, deriv=deriv)
                assert true_ip == approx(psi.value)
