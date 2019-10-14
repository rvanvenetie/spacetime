from ..datastructures.double_tree_view import DoubleTree
from ..space.triangulation import InitialTriangulation
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointBasis
from .basis import generate_y_delta


def test_full_tensor_y_delta():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    basis_space = triang.vertex_meta_root
    basis_space.uniform_refine(6)

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    # Tryout various levels.
    for l in range(6):
        # Create X^\delta
        X_delta = DoubleTree((basis_time.metaroot_wavelet, basis_space))
        X_delta.uniform_refine(l)

        # Generate Y^\delta
        Y_delta = generate_y_delta(X_delta)

        # Compare to the full tensor one.
        Y_delta_tensor = DoubleTree(
            (OrthonormalBasis.metaroot_wavelet, basis_space))
        Y_delta_tensor.uniform_refine(l)

        assert len(Y_delta.bfs()) == len(Y_delta_tensor.bfs())


def test_sparse_tensor_y_delta():
    # Create space part.
    triang = InitialTriangulation.unit_square()
    basis_space = triang.vertex_meta_root
    basis_space.uniform_refine(6)

    # Create time part for X^\delta
    basis_time = ThreePointBasis()
    basis_time.metaroot_wavelet.uniform_refine(6)

    # Refine the orthonormal basis
    OrthonormalBasis.metaroot_wavelet.uniform_refine(6)

    # Tryout various levels.
    for l in range(6):
        # Create X^\delta
        X_delta = DoubleTree((basis_time.metaroot_wavelet, basis_space))
        X_delta.sparse_refine(l)

        # Generate Y^\delta
        Y_delta = generate_y_delta(X_delta)

        # Compare to the sparse tensor one.
        Y_delta_tensor = DoubleTree(
            (OrthonormalBasis.metaroot_wavelet, basis_space))
        Y_delta_tensor.sparse_refine(l)

        assert len(Y_delta.bfs()) == len(Y_delta_tensor.bfs())
