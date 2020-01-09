from math import sqrt

import numpy as np

from ..datastructures.double_tree_view import DoubleTree
from ..spacetime.basis import generate_x_delta_underscore, generate_y_delta
from .heat_equation import HeatEquation
from .residual_error_estimator import ResidualErrorEstimator


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

        self.X_dd = None
        self.I_d_dd = None
        self.Y_dd = None

    def solve_step(self, x0=None, solver='pcg', tol=1e-5):
        info = {'dim_X_delta': len(self.X_delta.bfs())}
        print('\n\nAdaptive step for X_delta having {} nodes'.format(
            info['dim_X_delta']))

        self.X_dd, self.I_d_dd = generate_x_delta_underscore(self.X_delta)
        self.Y_dd = generate_y_delta(self.X_dd)
        info['dim_I_d_dd'] = len(self.I_d_dd)
        print('Number of funtions in X_delta_underscore \ X_delta {}'.format(
            len(self.I_d_dd)))

        # Calculate the solution using X_delta and Y_delta_hat.
        heat_dd_d = HeatEquation(X_delta=self.X_delta,
                                 Y_delta=self.Y_dd,
                                 formulation='schur',
                                 dirichlet_boundary=self.dirichlet_boundary)
        f_dd_d = heat_dd_d.calculate_rhs_vector(
            g_functional=self.g_functional, u0_functional=self.u0_functional)

        # If we have an initial guess, interpolate it into X_delta.
        if x0:
            x0 = x0.deep_copy()
            x0.union(self.X_delta, call_postprocess=None)

        u_dd_d, solve_info = heat_dd_d.solve(b=f_dd_d,
                                             x0=x0,
                                             solver=solver,
                                             tol=tol)
        info.update(solve_info)
        print('Solved in {} iterations.'.format(info['num_iters']))
        return u_dd_d, info

    def mark_refine(self, u_dd_d):
        error_estimator = ResidualErrorEstimator(
            u_dd_d=u_dd_d,
            g_functional=self.g_functional,
            u0_functional=self.u0_functional,
            X_d=self.X_delta,
            X_dd=self.X_dd,
            Y_dd=self.Y_dd,
            I_d_dd=self.I_d_dd,
            dirichlet_boundary=self.dirichlet_boundary)
        residual = error_estimator.res_dd_d
        residual_X_d = error_estimator.res_d
        residual_X_d_dd = error_estimator.res_dd_min_d
        info = {
            'res_norm': residual.norm(),
            'res_X_d_norm': np.linalg.norm([n.value for n in residual_X_d]),
            'res_X_d_dd_norm':
            np.linalg.norm([n.value for n in residual_X_d_dd]),
        }
        print('Residual norm is {}.'.format(info['res_norm']))

        marked_nodes = self.dorfler_marking(residual_X_d_dd, self.theta)
        info['n_marked'] = len(marked_nodes)
        print('Dorfler marked {} nodes'.format(len(marked_nodes)))

        # Create the new X_delta.
        self.X_delta = DoubleTree.make_conforming(residual_X_d + marked_nodes)
        info['dim_X_delta_ref'] = len(self.X_delta.bfs())
        print('X_delta grew from {} to {}.'.format(
            sum(not n.is_metaroot() for n in residual_X_d),
            info['dim_X_delta_ref']))

        return residual, info

    def solve(self, eps=1e-5, solver='pcg', max_iters=99999999):
        info = {
            'theta': self.theta,
            'eps': eps,
            'solver': solver,
            'max_iters': max_iters,
            'step_info': []
        }
        u_dd_d = None
        errors = []
        it = 0

        while True:
            it += 1
            u_dd_d, solve_info = self.solve_step(x0=u_dd_d, solver=solver)
            residual, mark_info = self.mark_refine(u_dd_d=u_dd_d)
            errors.append(mark_info['res_norm'])

            step_info = {}
            step_info.update(solve_info)
            step_info.update(mark_info)
            info['step_info'].append(step_info)

            if errors[-1] <= eps or it >= max_iters:
                break

        info['errors'] = errors
        info['n_steps'] = it
        info['converged'] = it < max_iters

        return u_dd_d, info

    @staticmethod
    def dorfler_marking(nodes, theta):
        vec_sqr = np.array([n.value for n in nodes])**2
        norm_sqr = vec_sqr.sum()
        sorted_indices = np.flip(np.argsort(vec_sqr))

        error_sqr = 0.0
        i = 0
        while error_sqr < theta**2 * norm_sqr and i < len(vec_sqr):
            error_sqr += vec_sqr[sorted_indices[i]]
            i += 1

        return [nodes[j] for j in sorted_indices[0:i]]
