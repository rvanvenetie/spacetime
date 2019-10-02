from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from ..datastructures.double_tree import DoubleTree
from .tree_plotter import TreePlotter


class DoubleTreePlotter:
    @staticmethod
    def plot_support_rectangles(doubletree, alpha=0.01):
        rects = []
        for node in doubletree.bfs():
            at, bt = node.nodes[0].support
            ax, bx = node.nodes[1].support
            rects.append(Rectangle((at, ax), bt - at, bx - ax))
        pc = PatchCollection(rects,
                             facecolor='r',
                             alpha=0.01,
                             edgecolor='None')
        fig, ax = plt.subplots(1)
        ax.add_collection(pc)

        return rects

    @staticmethod
    def plot_matplotlib_graph(doubletree, i_in):
        assert isinstance(doubletree, DoubleTree)
        import networkx as nx
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout
        from grave import plot_network
        from grave.style import use_attributes

        def onpick(ax, i, event):
            """ Event handler for clicking on a node. """
            if not hasattr(event, 'nodes') or not event.nodes:
                return

            graph = event.artist.graph
            # Reset previous highlighted node and highlight the current node.
            for node, attr in graph.nodes.data():
                attr.pop('color', None)
            double_node = event.nodes[0]
            graph.nodes[double_node]['color'] = 'C1'
            event.artist.stale = True
            event.artist.figure.canvas.draw_idle()

            # Update the right subplot to show the single-tree.
            ax[1].clear()
            TreePlotter.draw_matplotlib_graph(doubletree.fiber(
                not i, double_node.node),
                                              axis=ax[1])
            ax[1].set_title("Fiber of %s in axis %d" %
                            (double_node.node, not i))
            plt.draw()

        fig, axes = plt.subplots(2, 1)
        # Show the single tree in the left subplot.
        artist = TreePlotter.draw_matplotlib_graph(doubletree.project(i_in),
                                                   axis=axes[0])
        axes[0].set_title("Projection in axis %d" % i_in)

        # Add event handler for clicking on a node.
        artist.set_picker(10)
        fig.canvas.mpl_connect('pick_event', lambda x: onpick(axes, i_in, x))
        plt.draw()

    @staticmethod
    def plot_level_dots(doubletree):
        dots = defaultdict(int)
        for node_0 in doubletree.root.bfs(0):
            for node_1 in node_0.bfs(1):
                key = (node_0.nodes[0].level, node_1.nodes[1].level)
                dots[key] += 1
        ml0, ml1 = 0, 0
        for (l0, l1) in dots:
            ml0 = max(ml0, l0)
            ml1 = max(ml1, l1)
        dots_matrix = np.zeros((ml0 + 1, ml1 + 1))
        for (l0, l1) in dots:
            dots_matrix[l0, l1] = dots[(l0, l1)]
        plt.imshow(np.log(dots_matrix), origin='lower')
        plt.colorbar()

        return dots
