import numpy as np

from .heat_equation import HeatEquation


class ResidualErrorEstimator:
    """ Implements the residual error estimator of Proposition 5.7.

    Arguments:
        residual_vector: the doubletree vector `f_dd_dd - S_dd_dd u_dd_d`.
        X_d: the doubletree `X_delta`.
        X_dd: the doubletree `X_{underscore delta}`.
        I_d_dd: the nodes in `X_dd setminus X_d` in list format.
    """
    def __init__(self, residual_vector, X_d, X_dd, I_d_dd):
        # First, we will mark all the items that are in X_dd\X_d.
        for node in I_d_dd:
            node.marked = True

        # Also mark all the time wavelets that already existed in X_d.
        for node in X_d.project(0).bfs():
            node.node.marked = True

        # Return a list of the residual nodes on X_d and X_dd \ X_d.
        self.res_d = []
        self.res_dd_min_d = []

        def call_postprocess(res_node, other_node):
            # If this node is in X_d, the residual should be zero,
            # and there is nothing left to do.
            if not other_node.marked:
                assert abs(res_node.value) < 1e-5
                self.res_d.append(res_node)
                return
            else:
                self.res_dd_min_d.append(res_node)

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
        self.res_dd_d = residual_vector.deep_copy()
        self.res_dd_d.union(X_dd, call_postprocess=call_postprocess)

        # Unmark the nodes in X_dd \ X_d.
        for node in I_d_dd:
            node.marked = False

        # Also unmark all the time wavelets that exited in X_d.
        for node in X_d.project(0).bfs():
            node.node.marked = False

    @staticmethod
    def FromDoubleTrees(u_dd_d,
                        rhs_factory,
                        X_d,
                        X_dd,
                        Y_dd,
                        I_d_dd,
                        dirichlet_boundary=True):
        # First lift u_dd_d onto X_dd.
        u_dd_dd = u_dd_d.deep_copy()
        u_dd_dd.union(X_dd, call_postprocess=None)

        # Create operator/rhs for (X_dd, Y_dd).
        heat_dd_dd = HeatEquation(X_delta=X_dd,
                                  Y_delta=Y_dd,
                                  formulation='schur',
                                  dirichlet_boundary=dirichlet_boundary)
        f_dd_dd = rhs_factory(heat_dd_dd)

        # Calculate the residual wrt X_dd, i.e.  f_dd_dd - S_dd_dd(u_dd_d).
        residual_vector = f_dd_dd
        residual_vector -= heat_dd_dd.mat.apply(u_dd_dd)
        return ResidualErrorEstimator(residual_vector, X_d, X_dd, I_d_dd)
