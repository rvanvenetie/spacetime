from ..datastructures.double_tree_vector import (DoubleNodeVector,
                                                 DoubleTreeVector,
                                                 FrozenDoubleNodeVector)


class DoubleTreeFunction(DoubleTreeVector):
    """ Class that represents a function living on a double tree. """
    def __init__(self, root, frozen_dbl_cls=FrozenDoubleNodeVector):
        super().__init__(root=root, frozen_dbl_cls=frozen_dbl_cls)

    def eval(self, t, x, deriv=(False, False)):
        """ Evaluate in a stupid way. """
        if isinstance(t, np.array):
            result = np.zeros(t.shape) if not deriv[1] else np.zeros(x.shape)
        else:
            result = 0.0 if not deriv[1] else np.zeros(2)
        for node in self.bfs():
            result += node.nodes[0].eval(t, deriv[0]) * node.nodes[1].eval(
                x, deriv[1])
        return result

    @staticmethod
    def interpolant_of(dbl_tree, g, g_order):
        """ Interpolate `g` onto the span of `dbl_tree` using quadrature.

        Uses quadrature to find
            Interp(g) := sum_{lambda in dbl_tree} <g, phi_lambda> phi_lambda.

        Arguments:
            dbl_tree: the double tree to compute the projection on.
            g: a sum of separable functions: g == sum_i g_0i(t) * g_1i(x).
            g_order: tuple; maximum order of `g` in either axis.
        """
        def call_quad(nv, _):
            """ Helper function to do the quadrature for the input function. """
            if nv.is_metaroot(): return
            nv.value = sum(nv.nodes[0].inner_quad(g0, g_order=g_order[0]) *
                           nv.nodes[1].inner_quad(g1, g_order=g_order[1])
                           for g0, g1 in g)

        dbl_tree_fn = dbl_tree.deep_copy(mlt_node_cls=DoubleNodeVector,
                                         mlt_tree_cls=DoubleTreeFunction,
                                         call_postprocess=call_quad)
        return dbl_tree_fn
