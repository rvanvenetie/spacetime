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
                 u0_order,
                 u0_slice_norm,
                 dirichlet_boundary=True):
        super().__init__(g_functional, u0_functional, dirichlet_boundary)
        self.u0 = u0
        self.u0_slice_norm = u0_slice_norm
        self.u0_order = u0_order

    def estimate(
            self,
            heat_dd_d,
            u_dd_d,
            solver_tol=1e-5,
    ):
        Bu_minus_g = heat_dd_d.B.apply(u_dd_d)
        Bu_minus_g -= self.g_functional.eval(heat_dd_d.Y_delta)
        Bu_minus_g_array = Bu_minus_g.to_array()
        A_s = LinearOperatorApplicator(applicator=heat_dd_d.A_s,
                                       input_vec=Bu_minus_g)
        result, _ = A_s.solve(solver='pcg',
                              b=Bu_minus_g_array,
                              M=LinearOperatorApplicator(
                                  applicator=heat_dd_d.P_Y,
                                  input_vec=Bu_minus_g),
                              tol=solver_tol)
        u_dd_d_0 = u_dd_d.slice(i=0,
                                coord=0.0,
                                slice_cls=TriangulationFunction)
        zero_slice_error = u_dd_d_0.error_L2(
            lambda xy: self.u0(xy),
            self.u0_slice_norm,
            self.u0_order,
        )
        terms = (np.sqrt(Bu_minus_g_array.dot(result)), zero_slice_error)
        return np.sqrt(sum([t**2 for t in terms])), terms


class ResidualErrorEstimator(ErrorEstimator):
    @staticmethod
    def mean_zero_basis_transformation(vector):
        """ Transforms the space basis into a mean zero basis."""
        for db_node in reversed(vector.bfs()):
            if db_node.nodes[1].level == 0 or db_node.nodes[
                    1].on_domain_boundary or any(
                        parent.nodes[1].on_domain_boundary
                        for parent in db_node.parents[1]):
                continue
            for parent in db_node.parents[1]:
                db_node.value -= 0.5 * db_node.nodes[1].volume(
                ) / parent.nodes[1].volume() * parent.value

    def estimate(self, u_dd_d, X_d, X_dd, Y_dd, mean_zero=False):
        """ The residual error estimator of Proposition 5.7.

        Arguments:
            u_dd_d: the solution vector.
            X_d: the doubletree `X_delta`.
            X_dd: the doubletree `X_{underscore delta}`.
            Y_dd: the doubletree `Y_{underscore delta}`.
        """
        # First lift u_dd_d onto X_dd.
        u_dd_dd = u_dd_d.deep_copy()
        u_dd_dd.union(X_dd, call_postprocess=None)

        # Create operator/rhs for (X_dd, Y_dd).
        heat_dd_dd = HeatEquation(X_delta=X_dd,
                                  Y_delta=Y_dd,
                                  formulation='schur',
                                  dirichlet_boundary=self.dirichlet_boundary,
                                  use_space_cache=False)
        f_dd_dd = heat_dd_dd.calculate_rhs_vector(self.g_functional,
                                                  self.u0_functional)

        # Calculate the residual wrt X_dd, i.e.  f_dd_dd - S_dd_dd(u_dd_d).
        residual_vector = f_dd_dd
        residual_vector -= heat_dd_dd.mat.apply(u_dd_dd)

        # We mark all the nodes in X_d in X_dd.
        X_d_nodes = X_dd.union(X_d, call_filter=lambda _: False)
        for node in X_d_nodes:
            node.marked = True

        # Return a list of the residual nodes on X_d and X_dd \ X_d.
        res_d = []
        res_dd_min_d = []
        res_dd_d = residual_vector.deep_copy()

        # Do a basis transformation to mean zero space functions.
        if mean_zero:
            self.mean_zero_basis_transformation(res_dd_d)

        def calculate_residual(res_node, other_node):
            # There is nothing to do for metaroots.
            if other_node.is_metaroot(): return
            assert other_node.nodes[0].level >= 0 and other_node.nodes[
                1].level >= 0

            # If this node is in X_d, the residual should be zero,
            # and there is nothing left to do.
            if other_node.marked:
                res_d.append(res_node)
                return
            else:
                res_dd_min_d.append(res_node)

            # This is a node in X_dd \ X_d, we now evaluate the residual
            # using a scaled basis.
            lvl_diff = other_node.nodes[0].level - other_node.nodes[1].level
            res_node.value /= np.sqrt(1.0 + 4**(lvl_diff))

        # Calculate the residual.
        res_dd_d.union(X_dd, call_postprocess=calculate_residual)

        # Validate the boundary conditions.
        heat_dd_dd._validate_boundary_dofs(res_dd_d)

        # Unmark the nodes in X_dd
        for node in X_d_nodes:
            node.marked = False

        return res_dd_d, res_d, res_dd_min_d


class TimeSliceErrorEstimator(ErrorEstimator):
    """ This error estimator measures a time slice error in l2.

    This requires knowledge of the exact solution.
    """
    def __init__(self, u, u_order, u_slice_norm, dirichlet_boundary=True):
        super().__init__(g_functional=None,
                         u0_functional=None,
                         dirichlet_boundary=dirichlet_boundary)
        self.u = u
        self.u_slice_norm = u_slice_norm
        self.u_order = u_order

    def estimate(self, u_delta, times):
        errors = []
        for i, t in enumerate(times):
            sol_slice = u_delta.slice(i=0,
                                      coord=t,
                                      slice_cls=TriangulationFunction)
            errors.append(
                sol_slice.error_L2(
                    lambda xy: self.u[0](t) * self.u[1](xy),
                    self.u_slice_norm(t),
                    self.u_order[1],
                ))

        return np.array(errors)
