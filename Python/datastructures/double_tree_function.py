from ..datastructures.double_tree_vector import DoubleTreeVector


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
    def project_function(self, dbl_tree, g, g_order):
        def call_quad(nv, _):
            """ Helper function to do the quadrature for the input function. """
            if nv.is_metaroot(): return
            nv.value = sum(nv.nodes[0].inner_quad(fn0, fn_order=fn_order[0]) *
                           nv.nodes[1].inner_quad(fn1, fn_order=g_order[1])
                           for g0, g1 in g)

        dbl_tree_fn = dbl_tree.deep_copy(mlt_node_cls=DoubleNodeVector,
                                         mlt_tree_cls=DoubleTreeFunction,
                                         call_postprocess=call_quad)
