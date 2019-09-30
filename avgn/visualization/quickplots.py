import numpy as np
from avgn.visualization.projections import (
    scatter_projections,
    draw_projection_transitions,
)
from avgn.visualization.network_graph import plot_network_graph
import matplotlib.pyplot as plt


def draw_projection_plots(
    syllable_df,
    label_column="syllables_labels",
    projection_column="umap",
    figsize=(30, 10),
):
    """ draws three plots of transitions
    """
    fig, axs = plt.subplots(ncols=3, figsize=figsize)
    # plot scatter
    ax = axs[0]
    scatter_projections(
        projection=np.array(list(syllable_df[projection_column].values)),
        labels=syllable_df[label_column].values,
        ax=ax,
    )
    ax.axis("off")

    # plot transitions
    ax = axs[1]
    draw_projection_transitions(
        projections=np.array(list(syllable_df[projection_column].values)),
        sequence_ids=syllable_df["syllables_sequence_id"],
        sequence_pos=syllable_df["syllables_sequence_pos"],
        ax=ax,
    )
    ax.axis("off")

    # plot network graph
    ax = axs[2]
    elements = syllable_df[label_column].values
    projections = np.array(list(syllable_df[projection_column].values))
    sequence_ids = np.array(syllable_df["syllables_sequence_id"])
    plot_network_graph(
        elements, projections, sequence_ids, color_palette="tab20", ax=ax
    )

    ax.axis("off")

    return ax

