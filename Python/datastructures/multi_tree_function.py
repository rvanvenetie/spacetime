import numpy as np

from ..datastructures.multi_tree_vector import MultiTreeVector
from ..datastructures.double_tree_vector import DoubleTreeVector
from ..datastructures.tree_vector import TreeVector
from ..space.operators import Operator
from ..space.triangulation import (InitialTriangulation,
                                   to_matplotlib_triangulation)
from ..space.triangulation_view import TriangulationView


class MultiTreeFunction(MultiTreeVector):
    """ Class that represents a function living on a multi tree. """
    def eval(self, coords):
        """ Evaluate in a stupid way.
        
        Arguments:
            coords: a tuple of coords, each a float or array of size (dim, N).

        Returns:
            A float, or array of shape (N,).
        """
        assert isinstance(coords, tuple)
        assert len(coords) == self.root.dim
        # If the input coords are arrays, return an array, else a number.
        if any(isinstance(coord, np.ndarray) for coord in coords):
            assert all(coords[0].shape[-1] == coord.shape[-1]
                       for coord in coords)
            result = np.zeros(coords[0].shape[-1])
        else:
            result = 0.0

        for mlt_node in self.bfs():
            if mlt_node == 0.0: continue
            result += mlt_node.value * np.prod(
                [n.eval(c) for (n, c) in zip(mlt_node.nodes, coords)], axis=0)
        return result


class DoubleTreeFunction(MultiTreeFunction, DoubleTreeVector):
    def slice_time(self, t):
        """ Slices a double tree fn through a point in time. """
        result = TreeFunction.from_metaroot(self.root.nodes[1])

        for nv in self.project(0).bfs():
            # Check if t is contained inside support of time wavelet.
            a = float(nv.node.support[0].interval[0])
            b = float(nv.node.support[-1].interval[1])
            if a <= t <= b:
                result.axpy(nv.frozen_other_axis(), nv.node.eval(t))

        return result


class TreeFunction(MultiTreeFunction, TreeVector):
    def eval(self, coords):
        if not isinstance(coords, tuple): coords = (coords, )
        return super().eval(coords)

    def plot(self, fig=None, show=True, dirichlet_boundary=True):
        # Calculate the triangulation that is associated to the result.
        triang = TriangulationView(self)

        # Convert the result to single scale.
        space_operator = Operator(triang, dirichlet_boundary)
        self_ss = space_operator.apply_T(self.to_array())

        # Plot the result
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        matplotlib_triang = to_matplotlib_triangulation(
            triang.elem_tree_view, self)
        fig = fig or plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(matplotlib_triang, Z=self_ss)
        if show:
            plt.show()
