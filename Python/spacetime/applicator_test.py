from collections import defaultdict
from itertools import product
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from ..datastructures.double_tree import DoubleTree
from ..datastructures.double_tree_test import (corner_index_tree,
                                               full_tensor_double_tree,
                                               sparse_tensor_double_tree,
                                               uniform_index_tree)
from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 FrozenDoubleNodeVector)
from ..datastructures.function_test import FakeHaarFunction
from ..datastructures.tree_view import MetaRootView
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.applicator_inplace import Applicator as Applicator1D
from ..time.applicator_test import applicator_to_matrix
from ..time.haar_basis import HaarBasis
from ..time.operators import mass
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis
from .applicator import Applicator


class FakeApplicator(Applicator):
    class FakeSingleApplicator:
        def __init__(self, axis):
            self.axis = axis

        def apply(self, vec_in, vec_out):
            """ This simply sets all output values to 1. """
            for labda in vec_out.bfs():
                labda.dbl_node.value = 1

        def apply_low(self, vec_in, vec_out):
            """ This simply sets all output values to 1. """
            for labda in vec_out.bfs():
                labda.dbl_node.value = 1

        def apply_upp(self, vec_in, vec_out):
            """ This simply sets all output values to 1. """
            for labda in vec_out.bfs():
                labda.dbl_node.value = 1

    def __init__(self, Lambda_in, Lambda_out):
        super().__init__(Lambda_in=Lambda_in,
                         Lambda_out=Lambda_out,
                         applicator_time=self.FakeSingleApplicator('t'),
                         applicator_space=self.FakeSingleApplicator('x'))


class FakeHaarFunctionExt(FakeHaarFunction):
    """ Extend the FakeHaarFunction by fields necessary for the applicator. """
    def __init__(self, labda, f_type, parents=None, children=None):
        super().__init__(labda, f_type, parents, children)
        self.Sigma_psi_out = []
        self.Theta_psi_in = False

    @property
    def support(self):
        return [self]


class MockApplicator(Applicator):
    def __init__(self, Lambda_in, Lambda_out, applicator_time,
                 applicator_space):
        super().__init__(Lambda_in, Lambda_out, applicator_time,
                         applicator_space)

    def sigma(self):
        sigma_root = DoubleNodeVector(nodes=(self.Lambda_in.root.nodes[0],
                                             self.Lambda_out.root.nodes[1]),
                                      value=0)
        sigma = DoubleTree(sigma_root, frozen_dbl_cls=FrozenDoubleNodeVector)
        sigma.project(0).union(self.Lambda_in.project(0))
        sigma.project(1).union(self.Lambda_out.project(1))

        for psi_in_labda_0 in sigma.project(0).bfs():
            psi_in_labda_0.frozen_other_axis().union(
                self.Lambda_out.project(1))
        sigma.compute_fibers()
        return sigma

    def theta(self):
        theta_root = DoubleNodeVector(nodes=(self.Lambda_out.root.nodes[0],
                                             self.Lambda_in.root.nodes[1]),
                                      value=0)
        theta = DoubleTree(theta_root, frozen_dbl_cls=FrozenDoubleNodeVector)
        theta.project(0).union(self.Lambda_out.project(0))
        theta.project(1).union(self.Lambda_in.project(1))
        for psi_in_labda_1 in theta.project(1).bfs():
            psi_in_labda_1.frozen_other_axis().union(
                self.Lambda_out.project(0))
        theta.compute_fibers()
        return theta


def test_small_sigma():
    """ I computed on a piece of paper what Sigma should be for this combo. """
    root_time = uniform_index_tree(1, 't', node_class=FakeHaarFunctionExt)
    root_space = uniform_index_tree(1, 'x', node_class=FakeHaarFunctionExt)
    Lambda_in = full_tensor_double_tree(root_time, root_space)
    Lambda_out = full_tensor_double_tree(root_time, root_space)
    applicator = FakeApplicator(Lambda_in, Lambda_out)
    sigma = applicator.sigma()
    assert [n.nodes[0].labda for n in sigma.bfs()] == [(0, 0), (0, 0), (0, 0)]
    assert [n.nodes[1].labda for n in sigma.bfs()] == [(0, 0), (1, 0), (1, 1)]


def test_theta_full_tensor():
    HaarBasis.metaroot_wavelet.uniform_refine(4)
    Lambda_in = DoubleTree.full_tensor(HaarBasis.metaroot_wavelet,
                                       HaarBasis.metaroot_wavelet)
    Lambda_out = DoubleTree.full_tensor(HaarBasis.metaroot_wavelet,
                                        HaarBasis.metaroot_wavelet)
    applicator = FakeApplicator(Lambda_in, Lambda_out)
    theta = applicator.theta()
    assert [d_node.nodes for d_node in theta.bfs()
            ] == [d_node.nodes for d_node in Lambda_in.bfs()]


def test_theta_small():
    HaarBasis.metaroot_wavelet.uniform_refine(3)
    Lambda_in = DoubleTree(
        (HaarBasis.metaroot_wavelet, HaarBasis.metaroot_wavelet))
    Lambda_out = DoubleTree(
        (HaarBasis.metaroot_wavelet, HaarBasis.metaroot_wavelet))

    # Define the maximum levels
    lvls = (3, 1)

    # Refine Lambda_in in time towards the origin, while maintining the
    # maximum levels defined in lvls.
    #
    # This *magic* command creates a doubletree, with in the time axis a time
    # tree that is refined towards the origin, and fully tensored with a space
    # tree that is refined upto lvl[1].
    Lambda_in.deep_refine(
        call_filter=lambda d_node: d_node[0].level <= lvls[0] and d_node[1].
        level <= lvls[1] and (d_node[0].level < 1 or d_node[0].labda[1] == 0))

    # Same as Lambda_in but now refined towards the end of the unit interval.
    Lambda_out.deep_refine(call_filter=lambda d_node: d_node[0].level <= lvls[
        0] and d_node[1].level <= lvls[1] and (d_node[0].level < 1 or d_node[
            0].labda[1] == 2**(d_node[0].labda[0] - 1) - 1))

    assert len(Lambda_in.bfs()) == len(Lambda_out.bfs())
    applicator = FakeApplicator(Lambda_in, Lambda_out)
    theta = applicator.theta()

    assert [d.node for d in theta.project(1).bfs()
            ] == [d.node for d in Lambda_in.project(1).bfs()]

    assert set(
        (n.nodes[0].labda, n.nodes[1].labda) for n in theta.bfs()) == set([
            ((0, 0), (0, 0)),
            ((0, 0), (1, 0)),
            ((1, 0), (0, 0)),
            ((1, 0), (1, 0)),
        ])


def test_sigma_combinations():
    """ I have very little intuition for what Sigma does, exactly, so I just
    made a bunch of weird combinations of Lambda_in and Lambda_out, made Sigma,
    and hoped that everything is all-right. """
    for (L_in, L_out) in product(product(range(0, 5), range(0, 5)),
                                 product(range(0, 5), range(0, 5))):
        Lmax = max(L_in[0], L_out[0]), max(L_in[1], L_out[1])
        for roots in product([
                uniform_index_tree(
                    Lmax[0], 't', node_class=FakeHaarFunctionExt),
                corner_index_tree(Lmax[0], 't', node_class=FakeHaarFunctionExt)
        ], [
                uniform_index_tree(
                    Lmax[1], 'x', node_class=FakeHaarFunctionExt),
                corner_index_tree(Lmax[1], 'x', node_class=FakeHaarFunctionExt)
        ]):
            Lambdas_in = [
                full_tensor_double_tree(*roots, L_in),
                sparse_tensor_double_tree(*roots, L_in[0])
            ]
            Lambdas_out = [
                full_tensor_double_tree(*roots, L_out),
                sparse_tensor_double_tree(*roots, L_out[0])
            ]
            for (Lambda_in, Lambda_out) in product(Lambdas_in, Lambdas_out):
                applicator = FakeApplicator(Lambda_in, Lambda_out)
                sigma = applicator.sigma()
                for node in sigma.bfs():
                    assert all(
                        node.nodes[i].is_full() or node.nodes[i].is_leaf()
                        for i in [0, 1])


def test_applicator_real():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(2)

    # Create a hierarchical basis
    hierarch_basis = MetaRootView(metaroot=triang.vertex_meta_root,
                                  node_view_cls=HierarchicalBasisFunction)
    hierarch_basis.deep_refine()

    # Create time part for Lambda_in.
    HaarBasis.metaroot_wavelet.uniform_refine(2)

    # Create time part for Lambda_out.
    ThreePointBasis.metaroot_wavelet.uniform_refine(3)

    # Create Lambda_in/out and initialize the applicator.
    Lambda_in = DoubleTree.full_tensor(HaarBasis.metaroot_wavelet,
                                       hierarch_basis)
    Lambda_out = DoubleTree.full_tensor(ThreePointBasis.metaroot_wavelet,
                                        hierarch_basis)
    assert len(Lambda_in.bfs()) != len(Lambda_out.bfs())
    applicator = FakeApplicator(Lambda_in, Lambda_out)

    # Now create an vec_in and vec_out.
    vec_in = Lambda_in.deep_copy(dbl_node_cls=DoubleNodeVector,
                                 frozen_dbl_cls=FrozenDoubleNodeVector)
    vec_out = Lambda_out.deep_copy(dbl_node_cls=DoubleNodeVector,
                                   frozen_dbl_cls=FrozenDoubleNodeVector)

    assert len(vec_in.bfs()) == len(Lambda_in.bfs())
    assert all(n1.nodes == n2.nodes
               for n1, n2 in zip(vec_in.bfs(), Lambda_in.bfs()))

    # Initialize the input vector with ones.
    for db_node in vec_in.bfs():
        assert db_node.value == 0
        db_node.value = 1

    vec_out = applicator.apply(vec_in)
    assert all(d_node.value == 2 for d_node in vec_out.bfs())


def test_applicator_tensor_haar_mass():
    basis = HaarBasis()
    basis.metaroot_wavelet.uniform_refine(6)
    for l in range(1, 5):
        # Create Lambda_in/out and initialize the applicator.
        Lambda_in = DoubleTree(
            (HaarBasis.metaroot_wavelet, HaarBasis.metaroot_wavelet))
        Lambda_in.uniform_refine(l)
        Lambda_out = Lambda_in
        applicator_time = Applicator1D(mass(basis), basis_in=basis)
        applicator_space = Applicator1D(mass(basis), basis_in=basis)
        applicator = Applicator(Lambda_in, Lambda_out, applicator_time,
                                applicator_space)

        # First, calculate real matrix that corresponds to applicator.
        mat1d = np.diag([
            1 if psi.level == 0 else 2**(1 - psi.level)
            for psi in Lambda_in.project(0).bfs()
        ])
        mat2d = np.kron(mat1d, mat1d)

        def transform_dt_vector_to_np_vector(dt_vector):
            return np.array([
                psi_1.value for psi_0 in dt_vector.project(0).bfs()
                for psi_1 in psi_0.frozen_other_axis().bfs()
            ])

        # Test and apply 20 random vectors.
        for _ in range(20):
            # Initialze double tree vectors.
            vec_in = Lambda_in.deep_copy(dbl_node_cls=DoubleNodeVector,
                                         frozen_dbl_cls=FrozenDoubleNodeVector)

            assert len(vec_in.bfs()) == len(Lambda_in.bfs())
            assert all(n1.nodes == n2.nodes
                       for n1, n2 in zip(vec_in.bfs(), Lambda_in.bfs()))

            # Initialize the input vector with random numbers.
            for db_node in vec_in.bfs():
                assert db_node.value == 0
                db_node.value = np.random.rand()

            # Calculate the output.
            vec_out = applicator.apply(vec_in)

            # Transform the input/output vector to the mat2d coordinate format.
            tr_vec_in = transform_dt_vector_to_np_vector(vec_in)
            tr_vec_out = transform_dt_vector_to_np_vector(vec_out)

            # Calculate the result by plain old matvec, and compare!
            real_vec_out = mat2d.dot(tr_vec_in)
            assert np.allclose(real_vec_out, tr_vec_out)


def test_applicator_full_tensor_time():
    """ Takes a combination of wavelets on the time. """
    bases = [HaarBasis(), OrthonormalBasis(), ThreePointBasis()]
    for basis in bases:
        basis.metaroot_wavelet.uniform_refine(8)
    for basis_time, basis_space in product(bases, bases):
        basis.metaroot_wavelet.uniform_refine(8)
        print('\nTesting for basis_time={}, basis_space={}'.format(
            basis_time.__class__.__name__, basis_space.__class__.__name__))
        l_in = [3, 5]
        l_out = [2, 4]

        # Create Lambda_in/out and initialize the applicator.
        Lambda_in = DoubleTree(
            (basis_time.metaroot_wavelet, basis_space.metaroot_wavelet))
        Lambda_in.uniform_refine(l_in)
        print('\tLambda_in is tree upto levels {} with dofs {}'.format(
            l_in, len(Lambda_in.bfs())))
        Lambda_out = DoubleTree(
            (basis_time.metaroot_wavelet, basis_space.metaroot_wavelet))
        Lambda_out.uniform_refine(l_out)
        print('\tLambda_out is tree upto levels {} with dofs {}'.format(
            l_out, len(Lambda_out.bfs())))

        # Create 1D applicators
        applicator_time = Applicator1D(mass(basis_time), basis_in=basis_time)
        applicator_space = Applicator1D(mass(basis_space),
                                        basis_in=basis_space)
        applicator = Applicator(Lambda_in, Lambda_out, applicator_time,
                                applicator_space)

        # First, calculate real matrix that corresponds to applicator.
        mat_time = applicator_to_matrix(applicator_time, Lambda_in.project(0),
                                        Lambda_out.project(0))
        mat_space = applicator_to_matrix(applicator_space,
                                         Lambda_in.project(1),
                                         Lambda_out.project(1))
        mat2d = np.kron(mat_time, mat_space)

        def transform_dt_vector_to_np_vector(dt_vector):
            return np.array([
                psi_1.value for psi_0 in dt_vector.project(0).bfs()
                for psi_1 in psi_0.frozen_other_axis().bfs()
            ])

        # Test and apply 10 random vectors.
        for _ in range(10):
            # Initialze double tree vectors.
            vec_in = Lambda_in.deep_copy(dbl_node_cls=DoubleNodeVector,
                                         frozen_dbl_cls=FrozenDoubleNodeVector)

            assert len(vec_in.bfs()) == len(Lambda_in.bfs())
            assert all(n1.nodes == n2.nodes
                       for n1, n2 in zip(vec_in.bfs(), Lambda_in.bfs()))

            # Initialize the unit input vector.
            for db_node in vec_in.bfs():
                db_node.value = np.random.rand()

            # Calculate the output.
            vec_out = applicator.apply(vec_in)

            # Transform the input/output vector to the mat2d coordinate format.
            tr_vec_in = transform_dt_vector_to_np_vector(vec_in)
            tr_vec_out = transform_dt_vector_to_np_vector(vec_out)

            # Calculate the result by plain old matvec, and compare!
            real_vec_out = mat2d.dot(tr_vec_in)
            assert np.allclose(real_vec_out, tr_vec_out)


def test_applicator_different_out():
    hb = HaarBasis()
    ob = OrthonormalBasis()
    tp = ThreePointBasis()
    bases = [hb, ob, tp]
    for basis in bases:
        basis.metaroot_wavelet.uniform_refine(5)

    for basis_time_in, basis_space_in, basis_time_out, basis_space_out in [
        (hb, hb, tp, hb), (tp, tp, tp, hb), (tp, ob, hb, ob)
    ]:
        print('\nTesting for basis_time_in={}, basis_time_out={}'.format(
            basis_time_in.__class__.__name__,
            basis_time_out.__class__.__name__))
        print('Testing for basis_space_in={}, basis_space_out={}'.format(
            basis_space_in.__class__.__name__,
            basis_space_out.__class__.__name__))
        l_in = [3, 5]
        l_out = [2, 4]

        # Create Lambda_in/out and initialize the applicator.
        Lambda_in = DoubleTree(
            (basis_time_in.metaroot_wavelet, basis_space_in.metaroot_wavelet))
        Lambda_in.uniform_refine(l_in)
        print('\tLambda_in is tree upto levels {} with dofs {}'.format(
            l_in, len(Lambda_in.bfs())))
        Lambda_out = DoubleTree((basis_time_out.metaroot_wavelet,
                                 basis_space_out.metaroot_wavelet))
        Lambda_out.uniform_refine(l_out)
        print('\tLambda_out is tree upto levels {} with dofs {}'.format(
            l_out, len(Lambda_out.bfs())))

        # Create 1D applicators
        applicator_time = Applicator1D(mass(basis_time_in, basis_time_out),
                                       basis_in=basis_time_in,
                                       basis_out=basis_time_out)
        applicator_space = Applicator1D(mass(basis_space_in, basis_space_out),
                                        basis_in=basis_space_in,
                                        basis_out=basis_space_out)
        applicator = MockApplicator(Lambda_in, Lambda_out, applicator_time,
                                    applicator_space)

        # First, calculate real matrix that corresponds to applicator.
        mat_time = applicator_to_matrix(applicator_time, Lambda_in.project(0),
                                        Lambda_out.project(0))
        mat_space = applicator_to_matrix(applicator_space,
                                         Lambda_in.project(1),
                                         Lambda_out.project(1))
        mat2d = np.kron(mat_time, mat_space)

        def transform_dt_vector_to_np_vector(dt_vector):
            return np.array([
                psi_1.value for psi_0 in dt_vector.project(0).bfs()
                for psi_1 in psi_0.frozen_other_axis().bfs()
            ])

        # Test and apply 10 random vectors.
        for _ in range(10):
            # Initialze double tree vectors.
            vec_in = Lambda_in.deep_copy(dbl_node_cls=DoubleNodeVector,
                                         frozen_dbl_cls=FrozenDoubleNodeVector)

            assert len(vec_in.bfs()) == len(Lambda_in.bfs())
            assert all(n1.nodes == n2.nodes
                       for n1, n2 in zip(vec_in.bfs(), Lambda_in.bfs()))

            # Initialize the unit input vector.
            for db_node in vec_in.bfs():
                db_node.value = np.random.rand()

            # Calculate the output.
            vec_out = applicator.apply(vec_in)

            # Transform the input/output vector to the mat2d coordinate format.
            tr_vec_in = transform_dt_vector_to_np_vector(vec_in)
            tr_vec_out = transform_dt_vector_to_np_vector(vec_out)

            # Calculate the result by plain old matvec, and compare!
            real_vec_out = mat2d.dot(tr_vec_in)
            assert np.allclose(real_vec_out, tr_vec_out)
