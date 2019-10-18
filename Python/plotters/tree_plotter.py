import matplotlib.pyplot as plt


class TreePlotter:
    @staticmethod
    def draw_matplotlib_graph(root, axis=None, label_nodes=False):
        from networkx.drawing.nx_agraph import graphviz_layout
        from grave import plot_network
        from grave.style import use_attributes

        if axis is None:
            _, axis = plt.subplots(1, 1)
        G = _nx_graph_rooted_at(root)

        def font_styler(attributes):
            return {'font_size': 8, 'font_weight': .5, 'font_color': 'k'}

        return plot_network(
            G,
            ax=axis,
            node_style=use_attributes(),
            node_label_style=font_styler if label_nodes else None,
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
