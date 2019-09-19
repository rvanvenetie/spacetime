from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .double_tree import DoubleTree


class TreePlotter:
    def __init__(self, tree):
        self.tree = tree

    def plot_support_rectangles(self, alpha=0.01):
        rects = []
        for node in self.tree.bfs():
            at, bt = node.nodes[0].support
            ax, bx = node.nodes[1].support
            rects.append(Rectangle((at, ax), bt - at, bx - ax))
        pc = PatchCollection(rects,
                             facecolor='r',
                             alpha=0.01,
                             edgecolor='None')
        fig, ax = plt.subplots(1)
        ax.add_collection(pc)
        plt.show()

        return rects

    def plot_matplotlib_graph(self, i_in):
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
            new_G = nx_graph_rooted_at(double_node, i=not i)
            ax[1].clear()
            plot_network(new_G,
                         ax=ax[1],
                         layout=lambda x: graphviz_layout(new_G, prog='dot'))
            ax[1].set_title("Fiber of %s in axis %d" %
                            (double_node.nodes[i], not i))
            plt.draw()

        fig, axes = plt.subplots(2, 1)
        # Show the single tree in the left subplot.
        G = nx_graph_rooted_at(self.tree.root, i_in)
        axes[0].set_title("Projection in axis %d" % i_in)
        art0 = plot_network(G,
                            ax=axes[0],
                            node_style=use_attributes(),
                            layout=lambda x: graphviz_layout(G, prog='dot'))

        # Add event handler for clicking on a node.
        art0.set_picker(10)
        fig.canvas.mpl_connect('pick_event', lambda x: onpick(axes, i_in, x))
        plt.draw()

    def plot_level_dots(self):
        dots = defaultdict(int)
        for node_0 in self.tree.root.bfs(0):
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
        plt.show()

        return dots


def nx_graph_rooted_at(root, i):
    import networkx as nx
    G = nx.DiGraph()
    nodes = root.bfs(i)
    for node in nodes:
        G.add_node(node)
        for child in node.children[i]:
            G.add_edge(node, child)
    return G
