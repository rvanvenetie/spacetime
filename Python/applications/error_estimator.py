import numpy as np

from ..datastructures.applicator import LinearOperatorApplicator
from ..space.triangulation_function import TriangulationFunction
from .heat_equation import HeatEquation


class ErrorEstimator:
    """ An abstract error estimator.

    Arguments:
        g_functional: the functional of the data g.
        u0_functional: the functional of the data u0.
        dirichlet_boundary: whether or not to enforce Dirichlet BC.
    """
    def __init__(self, g_functional, u0_functional, dirichlet_boundary=True):
        self.g_functional = g_functional
        self.u0_functional = u0_functional
        self.dirichlet_boundary = dirichlet_boundary


class AuxiliaryErrorEstimator(ErrorEstimator):
    def __init__(self,
                 g_functional,
                 u0_functional,
                 u0,
                 u0_slice_norm,
                 u0_order,
                 dirichlet_boundary=True):
        self.g_functional = g_functional
        self.u0_functional = u0_functional
        self.u0 = u0
        self.u0_slice_norm = u0_slice_norm
        self.u0_order = u0_order
        self.dirichlet_boundary = dirichlet_boundary

    def estimate(self, u_dd_d, X_d, Y_dd, tol=1e-5):
        heat_dd_d = HeatEquation(X_delta=X_d,
                                 Y_delta=Y_dd,
                                 formulation='schur',
                                 dirichlet_boundary=self.dirichlet_boundary)
        Bu_minus_g = heat_dd_d.B.apply(u_dd_d)
        Bu_minus_g -= self.g_functional.eval(Y_dd)
        heat_dd_d.enforce_dirichlet_boundary(Bu_minus_g)
        Bu_minus_g_array = Bu_minus_g.to_array()
        A_s = LinearOperatorApplicator(applicator=heat_dd_d.A_s,
                                       input_vec=Bu_minus_g)
        result, _ = A_s.solve(solver='cg',
                              b=Bu_minus_g_array,
                              M=LinearOperatorApplicator(
                                  applicator=heat_dd_d.P_Y,
                                  input_vec=Bu_minus_g),
                              tol=tol)
        u_dd_d_0 = u_dd_d.slice(i=0,
                                coord=0.0,
                                slice_cls=TriangulationFunction)
        zero_slice_error = u_dd_d_0.error_L2(
            lambda xy: self.u0(xy),
            self.u0_slice_norm,
            self.u0_order,
        )
        return np.sqrt(Bu_minus_g_array.dot(result) + zero_slice_error**2)


class ResidualErrorEstimator(ErrorEstimator):
    def estimate(self, u_dd_d, X_d, X_dd, Y_dd, I_d_dd):
        """ The residual error estimator of Proposition 5.7.

        Arguments:
            u_dd_d: the solution vector.
            X_d: the doubletree `X_delta`.
            X_dd: the doubletree `X_{underscore delta}`.
            Y_dd: the doubletree `Y_{underscore delta}`.
            I_d_dd: the nodes in `X_dd setminus X_d` in list format.
        """
        # First lift u_dd_d onto X_dd.
        u_dd_dd = u_dd_d.deep_copy()
        u_dd_dd.union(X_dd, call_postprocess=None)

        # Create operator/rhs for (X_dd, Y_dd).
        heat_dd_dd = HeatEquation(X_delta=X_dd,
                                  Y_delta=Y_dd,
                                  formulation='schur',
                                  dirichlet_boundary=self.dirichlet_boundary)
        f_dd_dd = heat_dd_dd.calculate_rhs_vector(self.g_functional,
                                                  self.u0_functional)

        # Calculate the residual wrt X_dd, i.e.  f_dd_dd - S_dd_dd(u_dd_d).
        residual_vector = f_dd_dd
        residual_vector -= heat_dd_dd.mat.apply(u_dd_dd)

        # First, we will mark all the items that are in X_dd\X_d.
        for node in I_d_dd:
            node.marked = True

        # Also mark all the time wavelets that already existed in X_d.
        for node in X_d.project(0).bfs():
            node.node.marked = True

        # Return a list of the residual nodes on X_d and X_dd \ X_d.
        res_d = []
        res_dd_min_d = []

        def call_postprocess(res_node, other_node):
            # If this node is in X_d, the residual should be zero,
            # and there is nothing left to do.
            if not other_node.marked:
                assert abs(res_node.value) < 1e-5
                res_d.append(res_node)
                return
            else:
                res_dd_min_d.append(res_node)

            # There is nothing to do for metaroots.
            if other_node.is_metaroot(): return
            assert other_node.nodes[0].level >= 0 and other_node.nodes[
                1].level >= 0

            # This is a node in X_dd \ X_d, we now evaluate the residual
            # using a scaled basis.

            # This is a new time wavelet, scale with 2^(-|labda|)
            if not other_node.nodes[0].marked:
                assert other_node.nodes[1].level == 0
                res_node.value /= 2**(other_node.nodes[0].level)
            # This is a refined mesh function, scale with 1 + 4^|labda|-lvl(v).
            else:
                lvl_diff = other_node.nodes[0].level - other_node.nodes[1].level
                res_node.value /= np.sqrt(1.0 + 4**(lvl_diff))

        # Apply the basis transformation.
        res_dd_d = residual_vector.deep_copy()
        res_dd_d.union(X_dd, call_postprocess=call_postprocess)

        # Unmark the nodes in X_dd \ X_d.
        for node in I_d_dd:
            node.marked = False

        # Also unmark all the time wavelets that exited in X_d.
        for node in X_d.project(0).bfs():
            node.node.marked = False
        return res_dd_d, res_d, res_dd_min_d
