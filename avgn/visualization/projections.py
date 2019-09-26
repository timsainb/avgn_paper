import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib import collections as mc
import seaborn as sns


def scatter_projections(
    syllables=None,
    projection=None,
    labels=None,
    ax=None,
    figsize=(10, 10),
    alpha=0.1,
    s=1,
    color="k",
    color_palette="tab20",
):
    """ creates a scatterplot of syllables using some projection
    """
    if projection is None:
        if syllables is None:
            raise ValueError("Either syllables or projections must by passed")

        syllables_flattened = np.reshape(
            syllables, (np.shape(syllables)[0], np.prod(np.shape(syllables)[1:]))
        )

        # if no projection is passed, assume umap
        fit = umap.UMAP(min_dist=0.25, verbose=True)
        u_all = fit.fit_transform(syllables_flattened)

    # color labels
    if labels is not None:
        pal = sns.color_palette(color_palette, n_colors=len(np.unique(labels)))
        lab_dict = {lab: pal[i] for i, lab in enumerate(np.unique(labels))}
        colors = np.array([lab_dict[i] for i in labels])
    else:
        colors = color

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

        # plot
    ax.scatter(projection[:, 0], projection[:, 1], alpha=alpha, s=s, color=colors)
    return ax


def draw_projection_transitions(
    projections,
    sequence_ids,
    sequence_pos,
    ax=None,
    nseq=-1,
    cmap=plt.get_cmap("cubehelix"),
    alpha=0.05,
    linewidth=3,
):
    """ draws a line plot of each transition
    """
    # make a plot if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    for sequence in tqdm(np.unique(sequence_ids)):
        seq_mask = sequence_ids == sequence
        seq = sequence_pos[seq_mask]
        projection_seq = projections[seq_mask]
        colorline(
            projection_seq[:, 0],
            projection_seq[:, 1],
            ax,
            np.linspace(0, 1, len(projection_seq)),
            cmap=cmap,
            linewidth=linewidth,
            alpha=alpha,
        )
    minx, maxx, miny, maxy = (
        np.min(projections[:, 0]),
        np.max(projections[:, 0]),
        np.min(projections[:, 1]),
        np.max(projections[:, 1]),
    )
    ax.set_xlim((minx, maxx))
    ax.set_ylim((miny, maxy))
    return ax


def colorline(
    x,
    y,
    ax,
    z=None,
    cmap=plt.get_cmap("copper"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
):
    """ Plot a colored line with coordinates x and y
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_label_cluster_transitions(
    syllable_df,
    label_of_interest,
    superlabel="syllables_labels",
    sublabel="hdbscan_labels",
    projection_column="umap",
    line_alpha=0.01,
    scatter_alpha=0.1,
    color_palette="tab20",
    ax=None,
):
    """ Given a two sets of labels, plot the transitions 
    from one set of labels grouped by the second set of
    labels
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # subset the dataframe that is in the main cluster of interest
    subset_df = syllable_df[syllable_df[superlabel].values == label_of_interest]

    projections = np.array(list(syllable_df[projection_column].values))

    # unique labels and colors
    unique_labels = np.unique(subset_df[sublabel].values)

    # we make colors relative to all labels so this plot can match other plots
    # cpal = sns.color_palette(color_palette, len(unique_labels))
    all_labels = np.unique(syllable_df[sublabel].values)
    cpal = sns.color_palette(color_palette, len(all_labels))
    cpal_dict = {label: np.array(cpal)[all_labels == label] for label in unique_labels}
    # scatter background
    ax.scatter(
        projections[:, 0], projections[:, 1], color="k", alpha=scatter_alpha, s=1
    )

    # for every HDBSCAN label group in the subsetted dataframe
    for li, lab in enumerate(tqdm(unique_labels)):
        color = cpal_dict[lab]
        if lab == -1:
            continue
        # mask for only this label (orig + hdbscan)
        label_of_interest_mask = (
            syllable_df[superlabel].values == label_of_interest
        ) & (syllable_df[sublabel].values == lab)

        ax.scatter(
            projections[label_of_interest_mask][:, 0],
            projections[label_of_interest_mask][:, 1],
            s=1,
            color=color,
        )

        # DRAW OUTPUT FROM CLUSTER
        # TODO - ensure that inbound syllable_sequence_pos is not zero
        outbound = projections[label_of_interest_mask]
        inbound = projections[1:][label_of_interest_mask[:-1]]
        segments = [[i, j] for i, j in zip(outbound, inbound)]
        lc = mc.LineCollection(segments, colors=color, linewidths=1, alpha=line_alpha)
        ax.add_collection(lc)

        # DRAW INBOUND FROM CLUSTER
        outbound = projections[:-1][label_of_interest_mask[1:]]
        inbound = projections[label_of_interest_mask]
        segments = [[i, j] for i, j in zip(outbound, inbound)]
        lc = mc.LineCollection(segments, colors=color, linewidths=1, alpha=line_alpha)
        ax.add_collection(lc)

    return ax
