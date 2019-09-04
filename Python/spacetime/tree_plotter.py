from collections import defaultdict, deque
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np

from tree import DoubleTree


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
        def gen_G(root, i=None):
            G = nx.DiGraph()
            queue = deque()
            queue.append(root)
            nodes = []
            while queue:
                node = queue.popleft()
                if node.marked: continue
                G.add_node(node)
                if i is not None:
                    for child in node.children[i]:
                        G.add_edge(node, child)
                else:
                    for child in node.children:
                        G.add_edge(node, child)
                nodes.append(node)
                node.marked = True
                # Add the children to the queue.
                if i is not None:
                    queue.extend(node.children[i])
                else:
                    queue.extend(node.children)
            for node in nodes:
                node.marked = False
            return G

        def onpick(ax, i, event):
            graph = event.artist.graph
            if not hasattr(event, 'nodes') or not event.nodes:
                return
            for node, attr in graph.nodes.data():
                attr.pop('color', None)
            double_node = event.nodes[0]
            graph.nodes[double_node]['color'] = 'C1'
            event.artist.stale = True
            event.artist.figure.canvas.draw_idle()

            new_G = gen_G(double_node, i=1 - i)
            ax[1].clear()
            plot_network(new_G,
                         ax=ax[1],
                         layout=lambda x: graphviz_layout(new_G, prog='dot'))
            ax[1].set_title("Fiber in %s in axis %s" % (double_node, 1 - i))
            plt.draw()

        import networkx as nx
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout
        from grave import plot_network
        from grave.style import use_attributes

        fig, axes = plt.subplots(2, 1)
        G = gen_G(self.tree.root, i_in)
        axes[0].set_title("Double tree in axis %d" % i_in)
        art0 = plot_network(G,
                            ax=axes[0],
                            node_style=use_attributes(),
                            layout=lambda x: graphviz_layout(G, prog='dot'))
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
