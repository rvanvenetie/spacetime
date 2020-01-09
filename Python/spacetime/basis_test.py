from ..datastructures.double_tree_view import DoubleTree
from ..space.basis import HierarchicalBasisFunction
from ..space.triangulation import InitialTriangulation
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis
from .basis import generate_x_delta_underscore, generate_y_delta


def test_full_tensor_y_delta():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(6)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    # Tryout various levels.
    for l in range(6):
        # Create X^\delta
        X_delta = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        X_delta.uniform_refine(l)

        # Generate Y^\delta
        Y_delta = generate_y_delta(X_delta)

        # Compare to the full tensor one.
        Y_delta_tensor = DoubleTree.from_metaroots(
            (OrthonormalBasis.metaroot_wavelet, basis_space.root))
        Y_delta_tensor.uniform_refine(l)

        assert len(Y_delta.bfs()) == len(Y_delta_tensor.bfs())


def test_sparse_tensor_y_delta():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(6)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    # Refine the orthonormal basis
    OrthonormalBasis.metaroot_wavelet.uniform_refine(6)

    # Tryout various levels.
    for l in range(6):
        # Create X^\delta
        X_delta = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        X_delta.sparse_refine(l)

        # Generate Y^\delta
        Y_delta = generate_y_delta(X_delta)

        # Compare to the sparse tensor one.
        Y_delta_tensor = DoubleTree.from_metaroots(
            (OrthonormalBasis.metaroot_wavelet, basis_space.root))
        Y_delta_tensor.sparse_refine(l)

        assert len(Y_delta.bfs()) == len(Y_delta_tensor.bfs())


def test_x_delta_underscore():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(1)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.uniform_refine(0)
    assert len(X_delta.bfs()) == 8
    X_delta_underscore, I_delta = generate_x_delta_underscore(X_delta)
    assert len(X_delta_underscore.bfs()) == 22
    assert len(I_delta) == 22 - 8

    # Tryout various levels.
    for max_level, refine_lambda in [
        (6, lambda X, k: X.uniform_refine([k, 2 * k])),
        (8, lambda X, k: X.sparse_refine(k, weights=[2, 1]))
    ]:
        for l in range(1, max_level):
            # Create X^\delta
            X_delta = DoubleTree.from_metaroots(
                (basis_time.metaroot_wavelet, basis_space.root))
            refine_lambda(X_delta, l)
            X_delta_underscore, I_delta = generate_x_delta_underscore(X_delta)

            X_delta_refined = DoubleTree.from_metaroots(
                (basis_time.metaroot_wavelet, basis_space.root))
            refine_lambda(X_delta_refined, l + 2)

            assert all(dblnode.nodes[0].level <= (l + 1)
                       and dblnode.nodes[1].level <= (2 * (l + 1))
                       for dblnode in X_delta_underscore.bfs())
            N_Xd = len(X_delta.bfs())
            N_Xdu = len(X_delta_underscore.bfs())
            N_Xdr = len(X_delta_refined.bfs())
            assert N_Xd < N_Xdu <= N_Xdr
            assert len(I_delta) == N_Xdu - N_Xd


def test_x_delta_underscore_equal_to_sparse_grid():
    max_level = 5

    # Create space part.
    triang = InitialTriangulation.unit_square()
    triang.elem_meta_root.uniform_refine(2 * max_level)
    basis_space = HierarchicalBasisFunction.from_triangulation(triang)
    basis_space.deep_refine()

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(max_level)
    X_delta = DoubleTree.from_metaroots(
        (basis_time.metaroot_wavelet, basis_space.root))
    X_delta.uniform_refine(0)

    # Tryout various levels.
    for l in range(1, max_level):
        # Create X^\delta
        X_delta = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        X_delta.sparse_refine(2 * l, weights=[2, 1])
        X_delta_underscore, I_delta = generate_x_delta_underscore(X_delta)

        X_delta_refined = DoubleTree.from_metaroots(
            (basis_time.metaroot_wavelet, basis_space.root))
        X_delta_refined.sparse_refine(2 * (l + 1), weights=[2, 1])

        assert len(X_delta_refined.bfs()) == len(X_delta_underscore.bfs())
