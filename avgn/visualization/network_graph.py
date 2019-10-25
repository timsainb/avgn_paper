import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook as cb
from matplotlib.colors import colorConverter, Colormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import numpy as np
from networkx.utils import is_string_like
from tqdm.autonotebook import tqdm
import networkx as nx
import seaborn as sns


def plot_network_graph(
    elements,
    projections,
    sequence_ids,
    color_palette="tab20",
    ax=None,
    min_cluster_samples=0,
    pal_dict=None,
):
    """
    """
    sequences = [elements[sequence_ids == i] for i in np.unique(sequence_ids)]

    # compute the centers of each label
    element_centers = cluster_centers(elements, projections)

    # create a transition matrix of elements
    transition_matrix, element_dict, drop_list = build_transition_matrix(
        sequences, min_cluster_samples=min_cluster_samples
    )

    element_dict_r = {v: k for k, v in element_dict.items()}

    true_label = []
    used_drop_list = []
    for i in range(len(transition_matrix)):
        for drop in drop_list:
            if i + len(used_drop_list) >= drop:
                if drop not in used_drop_list:
                    used_drop_list.append(drop)
        true_label.append(i + len(used_drop_list))

    # generate graph
    graph = compute_graph(
        transition_matrix, min_connections=0.05, column_names=true_label
    )

    # graph positions
    pos = nx.random_layout(graph)
    pos = {i: element_centers[element_dict_r[i]] for i in pos.keys()}

    # get the palette
    if pal_dict is None:
        label_palette = sns.color_palette(color_palette, len(np.unique(elements)))
        pos_colors = [label_palette[i] for i in list(pos.keys())]
    else:
        pos_colors = [pal_dict[element_dict_r[i]] for i in pos.keys()]

    # get locations
    pos_locs = np.vstack(pos.values())

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    graph_weights = [graph[edge[0]][edge[1]]["weight"] for edge in graph.edges()]
    rgba_cols = [[0, 0, 0] + [i] for i in graph_weights]
    draw_networkx_edges(graph, pos, ax=ax, edge_color=rgba_cols, width=2)

    # centroids
    ax.scatter(
        pos_locs[:, 0],
        pos_locs[:, 1],
        color=np.array(pos_colors) * 0.5,
        s=250,
        zorder=100,
    )
    ax.scatter(pos_locs[:, 0], pos_locs[:, 1], color=pos_colors, s=150, zorder=100)

    return ax


def cluster_centers(elements, locations):
    """
    """
    return {
        element: np.mean(locations[elements == element], axis=0)
        for element in np.unique(elements)
    }


def build_transition_matrix(sequences, min_cluster_samples=0, ignore_elements=[-1]):
    """ builds a transition matrix from a set of sequences of discrete categories
    If there are fewer than min_cluster_samples samples for a given cluster,
     ignore it in the transition matrix
    """
    unique_elements = np.unique(np.concatenate(sequences))
    element_dict = {element: ei for ei, element in enumerate(unique_elements)}
    transition_matrix = np.zeros((len(unique_elements), len(unique_elements)))
    for sequence in tqdm(sequences):
        for si in range(len(sequence) - 1):
            transition_matrix[
                element_dict[sequence[si]], element_dict[sequence[si + 1]]
            ] += 1
    # breakme
    removed_columns = []
    for ri, row in enumerate(transition_matrix):

        if (np.sum(row) < min_cluster_samples) or (
            unique_elements[ri - len(removed_columns)] in ignore_elements
        ):

            transition_matrix = np.delete(
                transition_matrix, ri - len(removed_columns), 0
            )
            transition_matrix = np.delete(
                transition_matrix, ri - len(removed_columns), 1
            )
            unique_elements = np.delete(unique_elements, ri - len(removed_columns))
            removed_columns.append(ri)

    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=0)
    return transition_matrix, element_dict, removed_columns


def compute_graph(dense_matrix, min_connections=0.05, column_names=None):
    # Add all nodes to the list
    G = nx.DiGraph()
    # For each item in the node list get outgoing connection of the
    #   *last* item in the list from the dense array
    if column_names is None:
        column_names = np.arange(len(dense_matrix))

    for out_node in np.arange(len(dense_matrix)):
        in_list = np.array(np.where(dense_matrix[out_node] > min_connections)[0])
        for ini, in_node in enumerate(in_list):
            G.add_edge(column_names[out_node], column_names[in_node])
            G[column_names[out_node]][column_names[in_node]]["weight"] = dense_matrix[
                out_node
            ][in_node]
    return G


# modified from the networkx function
def draw_networkx_edges(
    G,
    pos,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=1.0,
    arrowstyle="-|>",
    arrowsize=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    arrows=True,
    label=None,
    node_size=300,
    nodelist=None,
    node_shape="o",
    rad=0.1,
    **kwds
):
    """Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    edgelist : collection of edge tuples
       Draw only specified edges(default=G.edges())

    width : float, or array of floats
       Line width of edges (default=1.0)

    edge_color : color string, or array of floats
       Edge color. Can be a single color format string (default='r'),
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)

    alpha : float
       The edge transparency (default=1.0)

    edge_ cmap : Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)

    edge_vmin,edge_vmax : floats
       Minimum and maximum for edge colormap scaling (default=None)

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
       Note: Arrows will be the same color as edges.

    arrowstyle : str, optional (default='-|>')
       For directed graphs, choose the style of the arrow heads.
       See :py:class: `matplotlib.patches.ArrowStyle` for more
       options.

    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head head's length and
       width. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute
       `mutation_scale` for more info.

    label : [None| string]
       Label for legend

    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edges

    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges

    Depending whether the drawing includes arrows or not.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False. Be sure to include `node_size` as a
    keyword argument; arrows are drawn considering the size of nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    if nodelist is None:
        nodelist = list(G.nodes())

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    if not cb.iterable(width):
        lw = (width,)
    else:
        lw = width

    if (
        not is_string_like(edge_color)
        and cb.iterable(edge_color)
        and len(edge_color) == len(edge_pos)
    ):
        if np.alltrue([is_string_like(c) for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c, alpha) for c in edge_color])
        elif np.alltrue([not is_string_like(c) for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if np.alltrue([cb.iterable(c) and len(c) in (3, 4) for c in edge_color]):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError("edge_color must contain color names or numbers")
    else:
        if is_string_like(edge_color) or len(edge_color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color, alpha),)
        else:
            msg = "edge_color must be a color or list of one color per edge"
            raise ValueError(msg)

    if not G.is_directed() or not arrows:
        edge_collection = LineCollection(
            edge_pos,
            colors=edge_colors,
            linewidths=lw,
            antialiaseds=(1,),
            linestyle=style,
            transOffset=ax.transData,
        )

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        # Note: there was a bug in mpl regarding the handling of alpha values
        # for each line in a LineCollection. It was fixed in matplotlib by
        # r7184 and r7189 (June 6 2009). We should then not set the alpha
        # value globally, since the user can instead provide per-edge alphas
        # now.  Only set it globally if provided as a scalar.
        if cb.is_numlike(alpha):
            edge_collection.set_alpha(alpha)

        if edge_colors is None:
            if edge_cmap is not None:
                assert isinstance(edge_cmap, Colormap)
            edge_collection.set_array(np.asarray(edge_color))
            edge_collection.set_cmap(edge_cmap)
            if edge_vmin is not None or edge_vmax is not None:
                edge_collection.set_clim(edge_vmin, edge_vmax)
            else:
                edge_collection.autoscale()
        return edge_collection

    arrow_collection = None
    if G.is_directed() and arrows:
        # Note: Waiting for someone to implement arrow to intersection with
        # marker.  Meanwhile, this works well for polygons with more than 4
        # sides and circle.

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head
        arrow_colors = edge_colors
        if arrow_colors is None:
            if edge_cmap is not None:
                assert isinstance(edge_cmap, Colormap)
            else:
                edge_cmap = plt.get_cmap()  # default matplotlib colormap
            if edge_vmin is None:
                edge_vmin = min(edge_color)
            if edge_vmax is None:
                edge_vmax = max(edge_color)
            color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)

        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src
            x2, y2 = dst
            arrow_color = None
            line_width = None
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            if cb.iterable(node_size):  # many node sizes
                src_node, dst_node = edgelist[i]
                index_node = nodelist.index(dst_node)
                marker_size = node_size[index_node]
                shrink_target = to_marker_edge(marker_size, node_shape)
            else:
                shrink_target = to_marker_edge(node_size, node_shape)
            if arrow_colors is None:
                arrow_color = edge_cmap(color_normal(edge_color[i]))
            elif len(arrow_colors) > 1:
                arrow_color = arrow_colors[i]
            else:
                arrow_color = arrow_colors[0]
            if len(lw) > 1:
                line_width = lw[i]
            else:
                line_width = lw[0]

            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle="arc3, rad=" + str(rad),
                arrowstyle=arrowstyle,
                shrinkA=shrink_source,
                shrinkB=shrink_target,
                mutation_scale=mutation_scale,
                color=arrow_color,
                linewidth=line_width,
                zorder=1,
            )  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    return arrow_collection
