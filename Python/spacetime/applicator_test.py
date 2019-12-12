from itertools import product

import numpy as np
import pytest
import scipy.sparse as sp

from ..datastructures.linop import KroneckerLinearOperator
from ..datastructures.tree_view import TreeView
from ..datastructures.double_tree_vector import DoubleTreeVector
from ..datastructures.double_tree_view import DoubleTree
from ..datastructures.double_tree_view_test import (corner_index_tree,
                                                    full_tensor_double_tree,
                                                    sparse_tensor_double_tree,
                                                    uniform_index_tree)
from ..datastructures.function_test import FakeHaarFunction
from ..space.applicator import Applicator as Applicator2D
from ..space.basis import HierarchicalBasisFunction
from ..space.operators import MassOperator as Mass2D
from ..space.operators import StiffnessOperator as Stiff2D
from ..space.triangulation import InitialTriangulation
from ..space.triangulation_view import TriangulationView
from ..time.applicator import Applicator as Applicator1D
from ..time.haar_basis import HaarBasis
from ..time.operators import mass as Mass1D
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis
from .applicator import Applicator, BlockDiagonalApplicator


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
                         applicator_space=self.FakeSingleApplicator('xy'))


class FakeHaarFunctionExt(FakeHaarFunction):
    """ Extend the FakeHaarFunction by fields necessary for the applicator. """
    def __init__(self, labda, f_type, parents=None, children=None):
        super().__init__(labda, f_type, parents, children)
        self.Sigma_psi_out = []
        self.Theta_psi_in = False

    @property
    def support(self):
        return [self]


class ApplicatorFullSigmaTheta(Applicator):
    """ Applicator which replaces Sigma and Theta by full-grid doubletrees. """
    def __init__(self, Lambda_in, Lambda_out, applicator_time,
                 applicator_space):
        super().__init__(Lambda_in, Lambda_out, applicator_time,
                         applicator_space)

    def _initialize_sigma(self):
        sigma = DoubleTreeVector.from_metaroots(
            (self.Lambda_in.root.nodes[0], self.Lambda_out.root.nodes[1]))
        sigma.project(0).union(self.Lambda_in.project(0))
        sigma.project(1).union(self.Lambda_out.project(1))

        for psi_in_labda_0 in sigma.project(0).bfs():
            psi_in_labda_0.frozen_other_axis().union(
                self.Lambda_out.project(1))
        sigma.compute_fibers()
        return sigma

    def _initialize_theta(self):
        theta = DoubleTreeVector.from_metaroots(
            (self.Lambda_out.root.nodes[0], self.Lambda_in.root.nodes[1]))
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
    root_space = uniform_index_tree(1, 'xy', node_class=FakeHaarFunctionExt)
    Lambda_in = full_tensor_double_tree(root_time, root_space)
    Lambda_out = full_tensor_double_tree(root_time, root_space)
    applicator = FakeApplicator(Lambda_in, Lambda_out)
    sigma = applicator.sigma
    assert [n.nodes[0].labda for n in sigma.bfs()] == [(0, 0), (0, 0), (0, 0)]
    assert [n.nodes[1].labda for n in sigma.bfs()] == [(0, 0), (1, 0), (1, 1)]


def test_theta_full_tensor():
    HaarBasis.metaroot_wavelet.uniform_refine(4)
    Lambda_in = DoubleTree.from_metaroots(
        (HaarBasis.metaroot_wavelet, HaarBasis.metaroot_wavelet))
    Lambda_in.deep_refine()
    Lambda_out = DoubleTree.from_metaroots(
        (HaarBasis.metaroot_wavelet, HaarBasis.metaroot_wavelet))
    Lambda_out.deep_refine()

    applicator = FakeApplicator(Lambda_in, Lambda_out)
    assert [d_node.nodes for d_node in applicator.theta.bfs()
            ] == [d_node.nodes for d_node in Lambda_in.bfs()]


def test_theta_small():
    HaarBasis.metaroot_wavelet.uniform_refine(3)
    Lambda_in = DoubleTree.from_metaroots(
        (HaarBasis.metaroot_wavelet, HaarBasis.metaroot_wavelet))
    Lambda_out = DoubleTree.from_metaroots(
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

    assert [d.node for d in applicator.theta.project(1).bfs()
            ] == [d.node for d in Lambda_in.project(1).bfs()]

    assert set((n.nodes[0].labda, n.nodes[1].labda)
               for n in applicator.theta.bfs()) == set([
                   ((0, 0), (0, 0)),
                   ((0, 0), (1, 0)),
                   ((1, 0), (0, 0)),
                   ((1, 0), (1, 0)),
               ])


def test_sigma_combinations():
    """ I have very little intuition for what Sigma does, exactly, so I just
    made a bunch of weird combinations of Lambda_in and Lambda_out, made Sigma,
    and hoped that everything is all-right. """
    for (L_in, L_out) in product(product(range(0, 3), range(0, 3)),
                                 product(range(0, 3), range(0, 3))):
        Lmax = max(L_in[0], L_out[0]), max(L_in[1], L_out[1])
        for roots in product([
                uniform_index_tree(
                    Lmax[0], 't', node_class=FakeHaarFunctionExt),
                corner_index_tree(Lmax[0], 't', node_class=FakeHaarFunctionExt)
        ], [
                uniform_index_tree(
                    Lmax[1], 'xy', node_class=FakeHaarFunctionExt),
                corner_index_tree(
                    Lmax[1], 'xy', node_class=FakeHaarFunctionExt)
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
                for node in applicator.sigma.bfs():
                    assert all(
                        node.nodes[i].is_full() or node.nodes[i].is_leaf()
                        for i in [0, 1])


def test_applicator_real():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(5)
    hierarch_basis = HierarchicalBasisFunction.from_triangulation(triang)
    hierarch_basis.uniform_refine()

    # Create space applicator
    applicator_space = Applicator2D(Mass2D())

    # Create time part for Lambda_in/Lambda_out
    basis_in = HaarBasis()
    basis_out = ThreePointBasis()
    basis_in.metaroot_wavelet.uniform_refine(3)
    basis_out.metaroot_wavelet.uniform_refine(5)

    # Create time applicator
    applicator_time = Applicator1D(Mass1D(basis_in, basis_out),
                                   basis_in=basis_in,
                                   basis_out=basis_out)

    # Create Lambda_in as sparse tree
    Lambda_in = DoubleTree.from_metaroots(
        (basis_in.metaroot_wavelet, hierarch_basis.root))
    Lambda_in.uniform_refine(5)

    # Create Lambda_out as full tree.
    Lambda_out = DoubleTree.from_metaroots(
        (ThreePointBasis.metaroot_wavelet, hierarch_basis.root))
    Lambda_out.sparse_refine(4)
    assert len(Lambda_in.bfs()) != len(Lambda_out.bfs())

    applicator = Applicator(Lambda_in, Lambda_out, applicator_time,
                            applicator_space)

    # Now create an vec_in and vec_out.
    vec_in = Lambda_in.deep_copy(mlt_tree_cls=DoubleTreeVector)

    assert len(vec_in.bfs()) == len(Lambda_in.bfs())
    assert all(n1.nodes == n2.nodes
               for n1, n2 in zip(vec_in.bfs(), Lambda_in.bfs()))

    # Initialize the input vector with ones.
    for db_node in vec_in.bfs():
        assert db_node.value == 0
        db_node.value = 1

    applicator.apply(vec_in)


def test_applicator_tensor_haar_Mass1D():
    basis = HaarBasis()
    basis.metaroot_wavelet.uniform_refine(6)
    for l in range(1, 5):
        # Create Lambda_in/out and initialize the applicator.
        Lambda_in = DoubleTree.from_metaroots(
            (HaarBasis.metaroot_wavelet, HaarBasis.metaroot_wavelet))
        Lambda_in.uniform_refine(l)
        Lambda_out = Lambda_in
        applicator_time = Applicator1D(Mass1D(basis), basis_in=basis)
        applicator_space = Applicator1D(Mass1D(basis), basis_in=basis)
        applicator = Applicator(Lambda_in, Lambda_out, applicator_time,
                                applicator_space)

        # First, calculate real matrix that corresponds to applicator.
        mat1d = np.diag([
            1 if psi.level == 0 else 2**(1 - psi.level)
            for psi in Lambda_in.project(0).bfs()
        ])
        mat2d = np.kron(mat1d, mat1d)

        # Test and apply 20 random vectors.
        for _ in range(20):
            # Initialze double tree vectors.
            vec_in = Lambda_in.deep_copy(mlt_tree_cls=DoubleTreeVector)

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
            tr_vec_in = vec_in.to_array()
            tr_vec_out = vec_out.to_array()

            # Calculate the result by plain old matvec, and compare!
            real_vec_out = mat2d.dot(tr_vec_in)
            assert np.allclose(real_vec_out, tr_vec_out)


@pytest.mark.slow
def test_applicator_full_tensor_time():
    """ Takes a combination of wavelets on the time. """
    bases = [HaarBasis(), OrthonormalBasis(), ThreePointBasis()]
    for basis in bases:
        basis.metaroot_wavelet.uniform_refine(8)
    for basis_time, basis_space in product(bases, bases):
        print('\nTesting for basis_time={}, basis_space={}'.format(
            basis_time.__class__.__name__, basis_space.__class__.__name__))
        l_in = [3, 5]
        l_out = [2, 4]

        # Create Lambda_in/out and initialize the applicator.
        Lambda_in = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.metaroot_wavelet))
        Lambda_in.uniform_refine(l_in)
        print('\tLambda_in is tree upto levels {} with dofs {}'.format(
            l_in, len(Lambda_in.bfs())))
        Lambda_out = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.metaroot_wavelet))
        Lambda_out.uniform_refine(l_out)
        print('\tLambda_out is tree upto levels {} with dofs {}'.format(
            l_out, len(Lambda_out.bfs())))

        # Create 1D applicators
        applicator_time = Applicator1D(Mass1D(basis_time), basis_in=basis_time)
        applicator_space = Applicator1D(Mass1D(basis_space),
                                        basis_in=basis_space)
        applicator = Applicator(Lambda_in, Lambda_out, applicator_time,
                                applicator_space)

        # First, calculate real matrix that corresponds to applicator.
        mat_time = applicator_time.to_matrix(Lambda_in.project(0),
                                             Lambda_out.project(0))
        mat_space = applicator_space.to_matrix(Lambda_in.project(1),
                                               Lambda_out.project(1))
        mat2d = np.kron(mat_time, mat_space)

        # Test and apply 10 random vectors.
        for _ in range(10):
            # Initialze double tree vectors.
            vec_in = Lambda_in.deep_copy(mlt_tree_cls=DoubleTreeVector)

            assert len(vec_in.bfs()) == len(Lambda_in.bfs())
            assert all(n1.nodes == n2.nodes
                       for n1, n2 in zip(vec_in.bfs(), Lambda_in.bfs()))

            # Initialize the unit input vector.
            for db_node in vec_in.bfs():
                db_node.value = np.random.rand()

            # Calculate the output.
            vec_out = applicator.apply(vec_in)

            # Transform the input/output vector to the mat2d coordinate format.
            tr_vec_in = vec_in.to_array()
            tr_vec_out = vec_out.to_array()

            # Calculate the result by plain old matvec, and compare!
            real_vec_out = mat2d.dot(tr_vec_in)
            assert np.allclose(real_vec_out, tr_vec_out)


@pytest.mark.slow
def test_applicator_full_tensor_spacetime_quad():
    """ Takes a combination of wavelets on the time against space. """
    bases = [HaarBasis(), OrthonormalBasis(), ThreePointBasis()]
    for basis in bases:
        basis.metaroot_wavelet.uniform_refine(8)

    triang = InitialTriangulation.unit_square()
    hierarch_basis = triang.vertex_meta_root
    hierarch_basis.uniform_refine(6)

    for basis_time_in, basis_time_out, op_space, l_in, l_out in product(
            bases, bases, [
                Mass2D(dirichlet_boundary=True),
                Mass2D(dirichlet_boundary=False),
                Stiff2D(dirichlet_boundary=False),
                Stiff2D(dirichlet_boundary=False)
            ], [(5, 4), (4, 5)], [(4, 5), (2, 6)]):
        print('\nTesting for basis_time_in={}, basis_time_out={}, '
              'op_space={}, l_in={}, l_out={}'.format(
                  basis_time_in.__class__.__name__,
                  basis_time_out.__class__.__name__,
                  op_space.__class__.__name__, l_in, l_out))

        # Create Lambda_in/out and initialize the applicator.
        Lambda_in = DoubleTree.from_metaroots(
            (basis_time_in.metaroot_wavelet, hierarch_basis))
        Lambda_in.uniform_refine(l_in)
        print('\tLambda_in is tree upto levels {} with dofs {}'.format(
            l_in, len(Lambda_in.bfs())))
        Lambda_out = DoubleTree.from_metaroots(
            (basis_time_out.metaroot_wavelet, hierarch_basis))
        Lambda_out.uniform_refine(l_out)
        print('\tLambda_out is tree upto levels {} with dofs {}'.format(
            l_out, len(Lambda_out.bfs())))

        # Create 1D applicators
        applicator_time = Applicator1D(Mass1D(basis_time_in, basis_time_out),
                                       basis_in=basis_time_in,
                                       basis_out=basis_time_out)
        applicator_space = Applicator2D(op_space)
        applicator = Applicator(Lambda_in, Lambda_out, applicator_time,
                                applicator_space)

        # First, calculate real matrix that corresponds to applicator.
        mat_time = applicator_time.to_matrix(Lambda_in.project(0),
                                             Lambda_out.project(0))
        mat_space = applicator_space.to_matrix(Lambda_in.project(1),
                                               Lambda_out.project(1))
        mat2d = np.kron(mat_time, mat_space)

        # Test and apply 10 random vectors.
        for _ in range(10):
            # Initialze double tree vectors.
            vec_in = Lambda_in.deep_copy(mlt_tree_cls=DoubleTreeVector)

            assert len(vec_in.bfs()) == len(Lambda_in.bfs())
            assert all(n1.nodes == n2.nodes
                       for n1, n2 in zip(vec_in.bfs(), Lambda_in.bfs()))

            # Initialize the unit input vector.
            for db_node in vec_in.bfs():
                db_node.value = np.random.rand()

            # Calculate the output and convert to the mat2d coordinate format.
            vec_out = applicator.apply(vec_in)
            tr_vec_in = vec_in.to_array()
            tr_vec_out = vec_out.to_array()

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
        Lambda_in = DoubleTree.from_metaroots(
            (basis_time_in.metaroot_wavelet, basis_space_in.metaroot_wavelet))
        Lambda_in.uniform_refine(l_in)
        print('\tLambda_in is tree upto levels {} with dofs {}'.format(
            l_in, len(Lambda_in.bfs())))
        Lambda_out = DoubleTree.from_metaroots(
            (basis_time_out.metaroot_wavelet,
             basis_space_out.metaroot_wavelet))
        Lambda_out.uniform_refine(l_out)
        print('\tLambda_out is tree upto levels {} with dofs {}'.format(
            l_out, len(Lambda_out.bfs())))

        # Create 1D applicators
        applicator_time = Applicator1D(Mass1D(basis_time_in, basis_time_out),
                                       basis_in=basis_time_in,
                                       basis_out=basis_time_out)
        applicator_space = Applicator1D(Mass1D(basis_space_in,
                                               basis_space_out),
                                        basis_in=basis_space_in,
                                        basis_out=basis_space_out)
        applicator = Applicator(Lambda_in, Lambda_out, applicator_time,
                                applicator_space)

        # First, calculate real matrix that corresponds to applicator.
        mat_time = applicator_time.to_matrix(Lambda_in.project(0),
                                             Lambda_out.project(0))
        mat_space = applicator_space.to_matrix(Lambda_in.project(1),
                                               Lambda_out.project(1))
        mat2d = np.kron(mat_time, mat_space)

        # Test and apply 10 random vectors.
        for _ in range(10):
            # Initialze double tree vectors.
            vec_in = Lambda_in.deep_copy(mlt_tree_cls=DoubleTreeVector)

            assert len(vec_in.bfs()) == len(Lambda_in.bfs())
            assert all(n1.nodes == n2.nodes
                       for n1, n2 in zip(vec_in.bfs(), Lambda_in.bfs()))

            # Initialize the unit input vector.
            for db_node in vec_in.bfs():
                db_node.value = np.random.rand()

            # Calculate the output.
            vec_out = applicator.apply(vec_in)

            # Transform the input/output vector to the mat2d coordinate format.
            tr_vec_in = vec_in.to_array()
            tr_vec_out = vec_out.to_array()

            # Calculate the result by plain old matvec, and compare!
            real_vec_out = mat2d.dot(tr_vec_in)
            assert np.allclose(real_vec_out, tr_vec_out)


def test_applicator_sparse_grid_time():
    """ Takes a combination of wavelets on the time. """
    bases = [HaarBasis(), OrthonormalBasis(), ThreePointBasis()]
    for basis in bases:
        basis.metaroot_wavelet.uniform_refine(5)
    for basis_time, basis_space in product(bases, bases):
        print('\nTesting for basis_time={}, basis_space={}'.format(
            basis_time.__class__.__name__, basis_space.__class__.__name__))
        l_in = 3
        l_out = 4

        # Create Lambda_in/out and initialize the applicator.
        Lambda_in = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.metaroot_wavelet))
        Lambda_in.sparse_refine(l_in)
        print('\tLambda_in is a sparse grid tree upto level {} with dofs {}'.
              format(l_in, len(Lambda_in.bfs())))
        Lambda_out = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.metaroot_wavelet))
        Lambda_out.sparse_refine(l_out)
        print('\tLambda_out is a sparse grid tree upto level {} with dofs {}'.
              format(l_out, len(Lambda_out.bfs())))

        # Create 1D applicators
        applicator_time = Applicator1D(Mass1D(basis_time), basis_in=basis_time)
        applicator_space = Applicator1D(Mass1D(basis_space),
                                        basis_in=basis_space)
        applicator = Applicator(Lambda_in, Lambda_out, applicator_time,
                                applicator_space)

        # Create another applicator with theta/sigma being full trees.
        applicator_ts = ApplicatorFullSigmaTheta(Lambda_in, Lambda_out,
                                                 applicator_time,
                                                 applicator_space)

        # Test and apply 10 random vectors.
        for _ in range(10):
            # Init double tree vector with random values.
            vec_in = Lambda_in.deep_copy(mlt_tree_cls=DoubleTreeVector)
            for db_node in vec_in.bfs():
                db_node.value = np.random.rand()

            # Calculate the output with both applicators and compare.
            vec_out = applicator.apply(vec_in)
            vec_out_ts = applicator_ts.apply(vec_in)
            assert np.allclose(vec_out.to_array(), vec_out_ts.to_array())


def KroneckerLinearOperator(R1, R2):
    """ Create LinOp that applies kron(A,B)x without explicit construction. """
    N, K = R1.shape
    M, L = R2.shape

    def matvec(x):
        X = x.reshape(K, L)
        return R2.dot(R1.dot(X).T).T.reshape(-1)

    return LinearOperator(matvec=matvec, shape=(N * M, K * L))


def test_applicator_time_identity():
    # Create space part.
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)
    vertex_view = TreeView.from_metaroot(T.vertex_meta_root)
    vertex_view.deep_refine()
    T_view = TriangulationView(vertex_view)

    hierarch_basis = HierarchicalBasisFunction.from_triangulation(T)
    hierarch_basis.deep_refine()

    # Create space applicator
    mass = Mass2D(T_view)
    applicator_space = Applicator2D(mass)

    # Create time basis for Lambda_in/Lambda_out
    basis = ThreePointBasis()
    basis.metaroot_wavelet.uniform_refine(5)

    # Create Lambda_in as full tree.
    Lambda = DoubleTree.from_metaroots(
        (basis.metaroot_wavelet, hierarch_basis.root))
    Lambda.uniform_refine(5)

    applicator = BlockDiagonalApplicator(Lambda, applicator_space)
    matrix = KroneckerLinearOperator(sp.eye(len(basis.metaroot_wavelet.bfs())),
                                     mass.as_linear_operator())
    # Test and apply 10 random vectors.
    for _ in range(10):
        # Init double tree vector with random values.
        vec_in = Lambda.deep_copy(mlt_tree_cls=DoubleTreeVector)
        for db_node in vec_in.bfs():
            db_node.value = np.random.rand()

        # Calculate the output with both applicators and compare.
        vec_out = applicator.apply(vec_in)
        assert np.allclose(vec_out.to_array(), matrix @ vec_in.to_array())
