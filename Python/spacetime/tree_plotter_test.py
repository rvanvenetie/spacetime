from tree import *
from tree_test import uniform_index_tree, corner_refined_index_tree, full_tensor_double_tree, sparse_tensor_double_tree, random_double_tree
from tree_plotter import *
import pytest


def show_rectangle_plot():
    for dt_root in [
            full_tensor_double_tree(corner_refined_index_tree(6, 'time', 0),
                                    corner_refined_index_tree(6, 'space', 1)),
            sparse_tensor_double_tree(corner_refined_index_tree(6, 'time', 0),
                                      corner_refined_index_tree(6, 'space', 1),
                                      6),
            random_double_tree(uniform_index_tree(5, 'time'),
                               uniform_index_tree(5, 'space'),
                               prob=0.2),
    ]:
        treeplotter = TreePlotter(DoubleTree(dt_root))
        treeplotter.plot_support_rectangles()


def show_level_dots():
    for dt_root in [
            full_tensor_double_tree(uniform_index_tree(6, 'time'),
                                    uniform_index_tree(6, 'space')),
            sparse_tensor_double_tree(uniform_index_tree(6, 'time'),
                                      uniform_index_tree(6, 'space'), 6),
            random_double_tree(uniform_index_tree(6, 'time'),
                               uniform_index_tree(6, 'space'),
                               prob=0.3),
    ]:
        treeplotter = TreePlotter(DoubleTree(dt_root))
        treeplotter.plot_level_dots()


if __name__ == "__main__":
    show_level_dots()
