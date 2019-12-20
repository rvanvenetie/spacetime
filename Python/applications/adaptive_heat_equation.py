from pprint import pprint

import numpy as np

from ..datastructures.double_tree_view import DoubleTree
from ..spacetime.basis import generate_x_delta_underscore, generate_y_delta
from .heat_equation import HeatEquation


class AdaptiveHeatEquation:
    """ Class for solving the heat equation using an adaptive loop. """
    def __init__(self,
                 X_init,
                 g_functional,
                 u0_functional,
                 theta,
                 dirichlet_boundary=True):
        self.X_delta = X_init
        self.g_functional = g_functional
        self.u0_functional = u0_functional
        self.theta = theta
        self.dirichlet_boundary = dirichlet_boundary

    def solve_step(self, x0=None, solver='pcg'):
        X_dd, I_d_dd = generate_x_delta_underscore(self.X_delta)
        Y_dd = generate_y_delta(X_dd)
        print('Number of funtions in X_delta_underscore \ X_delta {}'.format(
            len(I_d_dd)))

        # Calculate the solution using X_delta and Y_delta_hat.
        heat_dd_d = HeatEquation(X_delta=self.X_delta,
                                 Y_delta=Y_dd,
                                 formulation='schur',
                                 dirichlet_boundary=self.dirichlet_boundary)
        f_dd_d = heat_dd_d.calculate_rhs_vector(
            g_functional=self.g_functional, u0_functional=self.u0_functional)

        # If we have an initial guess, interpolate it into X_dd.
        if x0:
            x0 = x0.deep_copy()
            x0.union(self.X_delta, call_postprocess=None)

        u_dd_d, num_iters = heat_dd_d.solve(b=f_dd_d, x0=x0, solver=solver)

        residual, residual_X_d, residual_X_d_dd = self.residual(u_dd_d=u_dd_d,
                                                                X_dd=X_dd,
                                                                Y_dd=Y_dd,
                                                                I_d_dd=I_d_dd)
        print('Solved in {} iterations. Residual norm is {}.'.format(
            num_iters, residual.norm()))

        marked_nodes = self.dorfler_marking(residual_X_d_dd, self.theta)
        print('Dorfler marked {} nodes'.format(len(marked_nodes)))

        # Create the new X_delta.
        self.X_delta = DoubleTree.make_conforming(residual_X_d + marked_nodes)
        print('X_delta grew from {} to {}'.format(
            len([n for n in residual_X_d if not n.is_metaroot()]),
            len(self.X_delta.bfs())))

        return u_dd_d, residual

    def solve(self, eps=1e-5, solver='pcg'):
        u_dd_d, residual = self.solve_step(solver=solver)
        errors = [residual.norm()]

        while errors[-1] > eps:
            u_dd_d, residual = self.solve_step(x0=u_dd_d, solver=solver)
            errors.append(residual.norm())

        return u_dd_d, errors

    @staticmethod
    def dorfler_marking(nodes, theta):
        np_vec_sqr = np.array([n.value for n in nodes])**2
        vec_norm_sqr = np_vec_sqr.sum()
        sorted_indices = np.flip(np.argsort(np_vec_sqr))

        error_sqr = 0.0
        i = 0
        while error_sqr < theta**2 * vec_norm_sqr and i < len(np_vec_sqr):
            error_sqr += np_vec_sqr[sorted_indices[i]]
            i += 1

        return [nodes[j] for j in sorted_indices[0:i]]

    def residual(self, u_dd_d, X_dd, Y_dd, I_d_dd):
        # First interpolate u_dd_d into X_dd.
        u_dd_dd = u_dd_d.deep_copy()
        u_dd_dd.union(X_dd, call_postprocess=None)

        # Create operator/rhs for (X_dd, Y_dd).
        heat_dd_dd = HeatEquation(X_delta=X_dd,
                                  Y_delta=Y_dd,
                                  formulation='schur',
                                  dirichlet_boundary=self.dirichlet_boundary)
        f_dd_dd = heat_dd_dd.calculate_rhs_vector(
            g_functional=self.g_functional, u0_functional=self.u0_functional)

        # Calculate the residual wrt X_dd, i.e.  f_dd_dd - S_dd_dd(u_dd_d).
        residual = f_dd_dd
        residual -= heat_dd_dd.mat.apply(u_dd_dd)

        # First, we will mark all the items that are in X_dd\X_d.
        for node in I_d_dd:
            node.marked = True

        # Also mark all the time wavelets that already existed in X_d.
        for node in self.X_delta.project(0).bfs():
            node.node.marked = True

        # Return a list of the residual nodes on X_d, and X_dd \ X_d
        residual_X_d = []
        residual_X_d_dd = []

        def call_postprocess(res_node, other_node):
            # This node is in X_d, res_node.value should be zero.
            if not other_node.marked:
                assert abs(res_node.value) < 1e-5
                residual_X_d.append(res_node)
                return
            residual_X_d_dd.append(res_node)

            # There is nothing to do for metaroots.
            if other_node.is_metaroot(): return
            assert other_node.nodes[0].level >= 0 and other_node.nodes[
                1].level >= 0

            # In the other case, we must multiply res_node.value with something.

            # This is a new time wavelet, scale with 2^(-|labda|)
            if not other_node.nodes[0].marked:
                assert other_node.nodes[1].level == 0
                res_node.value /= 2**(other_node.nodes[0].level)
            # This is a refined mesh function, scale with 1 + 4^|labda|-lvl(v).
            else:
                lvl_diff = other_node.nodes[0].level - other_node.nodes[1].level
                res_node.value /= (1.0 + 4**(lvl_diff))

        # Apply the basis transformation
        residual.union(X_dd, call_postprocess=call_postprocess)

        # Do the unmarking.
        for node in I_d_dd:
            node.marked = False

        # Also mark all the time wavelets that already existed in X_d.
        for node in self.X_delta.project(0).bfs():
            node.node.marked = False

        # Return the residual
        return residual, residual_X_d, residual_X_d_dd
