import matplotlib.pyplot as plt

from .tree import NodeInterface


class TreePlotter:
    @staticmethod
    def draw_matplotlib_graph(root, axis=None):
        import networkx as nx
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout
        from grave import plot_network
        from grave.style import use_attributes

        if axis is None:
            _, axis = plt.subplots(1, 1)
        G = _nx_graph_rooted_at(root)
        return plot_network(G,
                            ax=axis,
                            node_style=use_attributes(),
                            layout=lambda x: graphviz_layout(G, prog='dot'))


def _nx_graph_rooted_at(root):
    import networkx as nx
    G = nx.DiGraph()
    nodes = root.bfs()
    for node in nodes:
        G.add_node(node)
        for child in node.children:
            G.add_edge(node, child)
    return G
