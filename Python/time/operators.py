from basis import (ContLinearScaling, DiscConstScaling, HaarBasis,
                        OrthoBasis, ThreePointBasis)
from linear_operator import LinearOperator


def _mass_haar_in_haar_out(phi_in):
    """ The scaling mass matrix for the Haar basis, 2**-l * Id. """
    l, n = phi_in.labda
    return [(phi_in, 2**-l)]


def _mass_ortho_in_ortho_out(phi_in):
    """ The scaling mass matrix for the orthonormal basis,  2**-l * Id. """
    l, n = phi_in.labda
    return [(phi_in, 2**-l)]


def _mass_three_in_three_out(phi_in):
    """ The scaling mass matrix for the three point basis (hat functions). """
    result = []
    l, n = phi_in.labda
    self_ip = 0
    if n > 0:
        assert phi_in.nbr_left
        result.append((phi_in.nbr_left, 1 / 6 * 2**-l))
        self_ip += 1 / 3 * 2**-l
    if n < 2**l:
        assert phi_in.nbr_right
        result.append((phi_in.nbr_right, 1 / 6 * 2**-l))
        self_ip += 1 / 3 * 2**-l
    result.append((phi_in, self_ip))
    return result


def _mass_haar_in_three_out(phi_in):
    """ The scaling mass matrix for haar in, three point out. """
    assert isinstance(phi_in, DiscConstScaling)
    elem = phi_in.support[0]
    assert elem.phi_cont_lin[0] and elem.phi_cont_lin[1]

    l, _ = phi_in.labda
    return [(elem.phi_cont_lin[0], 2**-(l + 1)),
            (elem.phi_cont_lin[1], 2**-(l + 1))]


def _mass_three_in_haar_out(phi_in):
    """ The scaling mass matrix for haar in, three point out. """
    assert isinstance(phi_in, ContLinearScaling)
    l, n = phi_in.labda
    result = []
    if n > 0:
        elem = phi_in.support[0]
        assert elem.phi_disc_const
        result.append((elem.phi_disc_const, 2**-(l + 1)))
    if n < 2**l:
        elem = phi_in.support[-1]
        assert elem.phi_disc_const
        result.append((elem.phi_disc_const, 2**-(l + 1)))
    return result


def mass(basis_in, basis_out=None):
    """ Returns the _scaling_ mass matrix corresponding to the given bases. """
    if basis_out is None:
        basis_out = basis_in

    if isinstance(basis_in, HaarBasis) and isinstance(basis_out, HaarBasis):
        return LinearOperator(_mass_haar_in_haar_out)
    elif isinstance(basis_in, OrthoBasis) and isinstance(
            basis_out, OrthoBasis):
        return LinearOperator(_mass_ortho_in_ortho_out)
    elif isinstance(basis_in, ThreePointBasis) and isinstance(
            basis_out, ThreePointBasis):
        return LinearOperator(_mass_three_in_three_out)
    elif isinstance(basis_in, HaarBasis) and isinstance(
            basis_out, ThreePointBasis):
        return LinearOperator(_mass_three_in_haar_out, _mass_haar_in_three_out)
    elif isinstance(basis_in, ThreePointBasis) and isinstance(
            basis_out, HaarBasis):
        return LinearOperator(_mass_haar_in_three_out, _mass_three_in_haar_out)
    else:
        raise TypeError(
            'Mass operator for ({}, {}) is not implemented (yet).'.format(
                basis_in.__class__.__name__, basis_out.__class__.__name__))
