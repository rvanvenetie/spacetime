from ..datastructures.double_tree_view import DoubleTree
from ..space.basis import HierarchicalBasisFunction
from ..time.orthonormal_basis import OrthonormalBasis
from ..time.three_point_basis import ThreePointWavelet


def generate_x_delta_underscore(x_delta):
    """ Generates X^{underline delta} as in p.27 from followup3.pdf. """
    assert isinstance(x_delta, DoubleTree)
    assert isinstance(x_delta.root.nodes[0].children[0], ThreePointWavelet)
    assert isinstance(x_delta.root.nodes[1].children[0],
                      HierarchicalBasisFunction)

    x_delta_underscore = x_delta.deep_copy()

    dblnodes = x_delta_underscore.bfs()
    time_leaves = []
    space_leaves = []
    for dblnode in dblnodes:
        # First, we ensure that the underlying trees are properly refined.
        if not dblnode.nodes[0].is_full():
            dblnode.nodes[0].refine()
        if not dblnode.nodes[1].node.is_full() or not dblnode.nodes[1].is_full(
        ):
            dblnode.nodes[1].node.refine()
            dblnode.nodes[1].refine(make_conforming=True)

        # The first part of this if-statement fends off the situation where
        # dblnode.children[i] has 0 < n < full elements as a result of adaptive
        # refinement in X_delta.
        if not dblnode.is_full(0):
            # Refine in time-axis...
            time_leaves.append(dblnode)

        if not dblnode.is_full(1):
            # and double-refine in space-axis.
            space_leaves.append(dblnode)

    for dblnode in time_leaves:
        dblnode.refine(i=0, make_conforming=True)

    for dblnode in space_leaves:
        children = dblnode.refine(i=1, make_conforming=True)
        for child in children:
            child.nodes[1].node.refine()
            child.nodes[1].refine(make_conforming=True)
            child.refine(i=1, make_conforming=True)

    dblnodes_underscore = x_delta_underscore.bfs()
    for dblnode in dblnodes:
        dblnode.marked = True

    new_dblnodes = []
    for dblnode in dblnodes_underscore:
        if dblnode.marked:
            dblnode.marked = False
        else:
            new_dblnodes.append(dblnode)

    return x_delta_underscore


def generate_y_delta(x_delta):
    """ Generates Y^\delta from X^\delta as p.6 from followup3.pdf. """

    assert isinstance(x_delta, DoubleTree)
    assert isinstance(x_delta.root.nodes[0].children[0], ThreePointWavelet)
    assert isinstance(x_delta.root.nodes[1].children[0],
                      HierarchicalBasisFunction)

    y_basis_time = OrthonormalBasis()

    # Create an empty double tree.
    y_delta = DoubleTree.from_metaroots(
        (y_basis_time.metaroot_wavelet, x_delta.root.nodes[1]))

    # Plug in the space metaroot axis.
    y_delta.project(1).union(x_delta.project(1))

    # For the time metaroot axis, we first *mark* which wavelets in time
    # should be tensored with which *trees* in space.
    for x_labda_0 in x_delta.project(0).bfs():
        for elem in x_labda_0.node.support:
            for mu in elem._refine_psi_orthonormal():
                if not mu.marked: mu.marked = []
                mu.marked.append(x_labda_0.frozen_other_axis())

    # First, we use this data to create the time metaroot axis.
    y_delta.project(0)._deep_refine(call_filter=lambda mu: mu.marked)

    # Then, we union the output wavelets with the right space trees.
    for y_labda_0 in y_delta.project(0).bfs():
        assert y_labda_0.node.marked
        for space_tree in y_labda_0.node.marked:
            y_labda_0.frozen_other_axis().union(space_tree)

        # Remove the marks.
        y_labda_0.node.marked = False

    y_delta.compute_fibers()

    return y_delta
