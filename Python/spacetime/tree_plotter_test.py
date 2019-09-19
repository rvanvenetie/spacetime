from .tree import *
from .tree_test import (uniform_index_tree, corner_index_tree,
                        full_tensor_double_tree, sparse_tensor_double_tree,
                        random_double_tree)
from .tree_plotter import *


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
        treeplotter = TreePlotter(DoubleTree(dt_root))
        treeplotter.plot_support_rectangles()


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
        treeplotter = TreePlotter(DoubleTree(dt_root))
        treeplotter.plot_matplotlib_graph(i_in=0)
        treeplotter.plot_matplotlib_graph(i_in=1)
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
        treeplotter = TreePlotter(DoubleTree(dt_root))
        print(treeplotter.plot_level_dots())


if __name__ == "__main__":
    show_matplotlib_graph()
