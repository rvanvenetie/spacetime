from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from mayavi import mlab

from tree import DoubleTree, bfs


class TreePlotter(object):
    def __init__(self, tree):
        self.tree = tree

    def plot_support_rectangles(self, alpha=0.01):
        rects = []
        for node in bfs(self.tree.root):
            at, bt = node.nodes[0].support
            ax, bx = node.nodes[1].support
            print((at, bt), (ax, bx))
            rects.append(Rectangle((at, ax), bt - at, bx - ax))
        pc = PatchCollection(rects,
                             facecolor='r',
                             alpha=0.01,
                             edgecolor='None')
        fig, ax = plt.subplots(1)
        ax.add_collection(pc)
        plt.show()

    def plot_mayavi_graph(self):
        G = nx.DiGraph()
        queue = deque()
        queue.append(self.tree.root)
        nodes = []
        while queue:
            node = queue.popleft()
            if node.marked: continue
            G.add_node("%s" % node)
            for child in node.children[0] + node.children[1]:
                G.add_edge("%s" % node, "%s" % child)
            nodes.append(node)
            node.marked = True
            # Add the children to the queue.
            queue.extend(node.children[0])
            queue.extend(node.children[1])

        for node in nodes:
            node.marked = False

        int_G = nx.convert_node_labels_to_integers(G)
        pos = nx.spring_layout(int_G, dim=3)
        xyz = np.array([pos[v] for v in sorted(int_G)])
        scalars = np.array(list(int_G.nodes())) + 5
        mlab.figure(1, bgcolor=(0, 0, 0))
        mlab.clf()

        pts = mlab.points3d(xyz[:, 0],
                            xyz[:, 1],
                            xyz[:, 2],
                            scalars,
                            scale_factor=0.1,
                            scale_mode='none',
                            colormap='Blues',
                            resolution=20)

        pts.mlab_source.dataset.lines = np.array(list(int_G.edges()))
        tube = mlab.pipeline.tube(pts, tube_radius=0.01)
        mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
        mlab.show()

    def plot_level_dots(self):
        dots = {}
        for node_0 in bfs(self.tree.root, 0):
            for node_1 in bfs(node_0, 1):
                key = (node_0.nodes[0].level, node_1.nodes[1].level)
                if key in dots:
                    dots[key] += 1
                else:
                    dots[key] = 1
        print(dots)
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
