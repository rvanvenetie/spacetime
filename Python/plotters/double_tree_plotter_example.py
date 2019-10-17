import matplotlib.pyplot as plt

from ..datastructures.double_tree_view import DoubleTree
from ..datastructures.double_tree_view_test import (corner_index_tree,
                                                    full_tensor_double_tree,
                                                    random_double_tree,
                                                    sparse_tensor_double_tree,
                                                    uniform_index_tree)
from ..space.triangulation import InitialTriangulation
from .double_tree_plotter import DoubleTreePlotter
from .tree_plotter import TreePlotter


def show_rectangle_plot():
    for dt_root in [
            full_tensor_double_tree(corner_index_tree(6, 't', 0),
                                    corner_index_tree(6, 'x', 1)),
            sparse_tensor_double_tree(corner_index_tree(6, 't', 0),
                                      corner_index_tree(6, 'x', 1), 6),
            random_double_tree(uniform_index_tree(7, 't'),
                               uniform_index_tree(7, 'x'),
                               7,
                               N=500),
    ]:
        DoubleTreePlotter.plot_support_rectangles(dt_root)
        plt.show()


def show_matplotlib_graph():
    for dt_root in [
            full_tensor_double_tree(uniform_index_tree(5, 't'),
                                    uniform_index_tree(5, 'x')),
            sparse_tensor_double_tree(uniform_index_tree(5, 't'),
                                      uniform_index_tree(5, 'x'), 5),
            random_double_tree(uniform_index_tree(7, 't'),
                               uniform_index_tree(7, 'x'),
                               7,
                               N=500),
    ]:
        DoubleTreePlotter.plot_matplotlib_graph(dt_root, i_in=0)
        DoubleTreePlotter.plot_matplotlib_graph(dt_root, i_in=1)
        plt.show()


def show_spacetime_tree():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(5)
    dt_root = full_tensor_double_tree(uniform_index_tree(5, 't'),
                                      T.vertex_meta_root)
    DoubleTreePlotter.plot_matplotlib_graph(dt_root, i_in=0)
    DoubleTreePlotter.plot_matplotlib_graph(dt_root, i_in=1)
    plt.show()


def show_level_dots():
    for dt_root in [
            full_tensor_double_tree(uniform_index_tree(6, 't'),
                                    uniform_index_tree(6, 'x')),
            sparse_tensor_double_tree(uniform_index_tree(6, 't'),
                                      uniform_index_tree(6, 'x'), 6),
            random_double_tree(uniform_index_tree(7, 't'),
                               uniform_index_tree(7, 'x'),
                               7,
                               N=500),
    ]:
        DoubleTreePlotter.plot_level_dots(dt_root)
        plt.show()


if __name__ == "__main__":
    show_spacetime_tree()
