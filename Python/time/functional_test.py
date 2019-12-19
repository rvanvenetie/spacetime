import itertools

from pytest import approx

from . import operators
from .functional import Functional
from .haar_basis import HaarBasis
from .orthonormal_basis import OrthonormalBasis
from .three_point_basis import ThreePointBasis


def test_functional_quadrature():
    uml = 4
    oml = 11
    hbu = HaarBasis.uniform_basis(max_level=1)
    hbo = HaarBasis.origin_refined_basis(max_level=oml)
    hbe = HaarBasis.end_points_refined_basis(max_level=oml)
    oru = OrthonormalBasis.uniform_basis(max_level=uml)
    oro = OrthonormalBasis.origin_refined_basis(max_level=oml)
    ore = OrthonormalBasis.end_points_refined_basis(max_level=oml)
    tpu = ThreePointBasis.uniform_basis(max_level=uml)
    tpo = ThreePointBasis.origin_refined_basis(max_level=oml)
    tpe = ThreePointBasis.end_points_refined_basis(max_level=oml)
    for basis, deriv in itertools.product(
        [hbu, hbo, hbe, oru, oro, ore, tpu, tpo, tpe], [False, True]):
        basis, Lambda = basis
        for g, g_order in [(lambda t: -2 * (1 + t**2), 2),
                           (lambda t: 2 * t, 1)]:
            operator = operators.quadrature(g=g, g_order=g_order, deriv=deriv)
            applicator = Functional(operator, basis)
            for _ in range(10):
                # Calculate <g, Phi>.
                inner_g_vec = applicator.eval(Lambda)

                # Compare
                for psi in Lambda:
                    true_ip = psi.inner_quad(g=g, g_order=g_order, deriv=deriv)
                    assert true_ip == approx(inner_g_vec[psi])
