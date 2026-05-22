import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def construir_grafo(model):
    """
    Construye una representación en grafo de un modelo de red neuronal.
    """
    G = nx.Graph()

    sizes = []

    # Extraer tamaños de capas
    for layer in model.layers:
        if hasattr(layer, "W"):
            in_features = layer.W.data.shape[0]
            out_features = layer.W.data.shape[1]

            if not sizes:
                sizes.append(in_features)

            sizes.append(out_features)

    # Crear nodos para cada capa
    node_id = 0
    layer_nodes = []

    for layer_idx, size in enumerate(sizes):
        nodes = []

        for _ in range(size):
            G.add_node(node_id, layer=layer_idx)
            nodes.append(node_id)
            node_id += 1

        layer_nodes.append(nodes)

    # Conectar capas
    for i in range(len(layer_nodes) - 1):
        for u in layer_nodes[i]:
            for v in layer_nodes[i + 1]:
                G.add_edge(u, v)

    return G


def dibujar_red(model):
    """
    Visualiza la arquitectura de una red neuronal.
    """
    G = construir_grafo(model)

    pos = nx.multipartite_layout(G, subset_key="layer")

    # Escalar posiciones de nodos
    layer_spacing = 3
    node_spacing = 1

    for node, (x, y) in pos.items():
        pos[node] = (x * layer_spacing, y * node_spacing)

    # Definir colores por capa
    colors = []
    max_layer = max(nx.get_node_attributes(G, "layer").values())

    for _, data in G.nodes(data=True):
        layer = data["layer"]

        if layer == 0:
            colors.append("limegreen")

        elif layer == max_layer:
            colors.append("red")

        else:
            colors.append("blue")

    # Dibujar red neuronal
    plt.figure(figsize=(12, 8))

    nx.draw(
        G,
        pos,
        node_color=colors,
        with_labels=False,
        node_size=80,
        width=0.125,
        alpha=1
    )

    plt.title("PoliFlow Neural Network")
    
    
    # =========================
    # 🔹 LEYENDA
    # =========================
    leyenda = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Entrada',
            markerfacecolor='limegreen',
            markersize=12
        ),

        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Neuronas de capas ocultas',
            markerfacecolor='blue',
            markersize=12
        ),

        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label='Salida',
            markerfacecolor='red',
            markersize=12
        )
    ]

    plt.legend(
        handles=leyenda,
        loc='upper right',
        frameon=True,
        fancybox=True,
        framealpha=1,
        borderpad=0.8,
        labelspacing=0.8
    )

    
    plt.axis("off")
    plt.show()
































"""
import networkx as nx
import matplotlib.pyplot as plt


def build_network_graph(model):
    G = nx.Graph()

    layers = []
    sizes = []

    # extraer tamaños de capas
    for layer in model.layers:
        if hasattr(layer, "W"):
            in_features = layer.W.data.shape[0]
            out_features = layer.W.data.shape[1]

            if not sizes:
                sizes.append(in_features)
            sizes.append(out_features)

    # crear nodos por capa
    node_id = 0
    layer_nodes = []

    for layer_idx, size in enumerate(sizes):
        nodes = []
        for _ in range(size):
            G.add_node(node_id, layer=layer_idx)
            nodes.append(node_id)
            node_id += 1
        layer_nodes.append(nodes)

    # conectar capas
    for i in range(len(layer_nodes) - 1):
        for u in layer_nodes[i]:
            for v in layer_nodes[i + 1]:
                G.add_edge(u, v)

    return G


def draw_network(model):
    G = build_network_graph(model)

    pos = nx.multipartite_layout(G, subset_key="layer")

    # =========================
    # 🔹 ESCALAR POSICIONES
    # =========================
    layer_spacing = 3   # ← separa capas (horizontal)
    node_spacing = 1    # ← separa neuronas (vertical)

    for node, (x, y) in pos.items():
        pos[node] = (x * layer_spacing, y * node_spacing)
        
    
    
    #for node, (x, y) in pos.items():
    #    pos = nx.multipartite_layout(G, subset_key="layer", scale=1)
    

    # =========================
    # 🔹 COLORES POR CAPA
    # =========================
    colors = []
    max_layer = max(nx.get_node_attributes(G, "layer").values())

    for _, data in G.nodes(data=True):
        layer = data["layer"]
        if layer == 0:
            colors.append("limegreen")          # input
        elif layer == max_layer:
            colors.append("red")   # output
        else:
            colors.append("blue")       # hidden

    # =========================
    # 🔹 DIBUJAR
    # =========================
    plt.figure(figsize=(12, 8))

    nx.draw(
        G,
        pos,
        node_color=colors,
        with_labels=False,
        node_size=80,     # tamaño de nodos
        width=0.125,        # ← líneas MÁS DELGADAS 🔥
        alpha=1         # transparencia (opcional)
    )

    plt.title("PoliFlow Neural Network")
    plt.axis("off")
    plt.show()
"""    
   
