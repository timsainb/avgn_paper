from scipy.cluster import hierarchy
from nltk.metrics.distance import edit_distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from joblib import Parallel, delayed
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.autonotebook import tqdm


def song_barcode(
    start_times, stop_times, labels, label_dict, label_pal_dict, resolution=0.01
):
    """ create a barcode for a single song rendition
    """
    begin = np.min(start_times)
    end = np.max(stop_times)
    trans_list = (
        np.zeros(int((end - begin) / resolution)).astype("str").astype("object")
    )
    # print(end, begin, end-begin, resolution, len(trans_list))
    for start, stop, label in zip(start_times, stop_times, labels):
        trans_list[
            int((start - begin) / resolution) : int((stop - begin) / resolution)
        ] = label_dict[label]

    color_list = [
        label_pal_dict[i] if i in label_pal_dict else [1, 1, 1] for i in trans_list
    ]
    color_list = np.expand_dims(color_list, 1)

    return trans_list, color_list


def indv_barcode(indv_df, time_resolution=0.02, label="labels", pal="tab20"):
    """ Create a barcode list for an individual
    """
    unique_labels = indv_df[label].unique()
    # song palette
    label_pal = np.random.permutation(sns.color_palette(pal, len(unique_labels)))
    label_dict = {lab: str(int(i)).zfill(3) for i, lab in enumerate(unique_labels)}

    label_pal_dict = {
        label_dict[lab]: color for lab, color in zip(unique_labels, label_pal)
    }
    sns.palplot(list(label_pal_dict.values()))

    # get list of syllables by time
    trans_lists = []
    color_lists = []
    for key in tqdm(indv_df.key.unique(), leave=False):
        # dataframe of wavs
        wav_df = indv_df[indv_df["key"] == key]
        labels = wav_df[label].values
        start_times = wav_df.start_time.values
        stop_times = wav_df.end_time.values
        start_times[:3], stop_times[:3], labels[:3]
        trans_list, color_list = song_barcode(
            start_times,
            stop_times,
            labels,
            label_dict,
            label_pal_dict,
            resolution=time_resolution,
        )
        color_lists.append(color_list)
        trans_lists.append(trans_list)

    return color_lists, trans_lists, label_pal_dict, label_pal, label_dict


def plot_sorted_barcodes(
    color_lists,
    trans_lists,
    max_list_len=600,
    seq_len=100,
    nex=50,
    n_jobs=-1,
    figsize=(25, 6),
    ax=None,
):
    """ cluster and plot barcodes sorted
    max_list_len = 600 # maximum length to visualize
    seq_len = 100 # maximumim length to compute lev distance
    nex = 50 # only show up to NEX examples
    """
    # subset dataset
    color_lists = color_lists[:nex]
    trans_lists = trans_lists[:nex]

    # get length of lists
    list_lens = [len(i) for i in trans_lists]

    # set max list length
    if max_list_len is None:
        max_list_len = np.max(list_lens)

    # make a matrix for color representations of syllables
    color_item = np.ones((max_list_len, len(list_lens), 3))
    for li, _list in enumerate(tqdm(color_lists, leave=False)):
        color_item[: len(_list), li, :] = np.squeeze(_list[:max_list_len])
    color_items = color_item.swapaxes(0, 1)

    # make a list of symbols padded to equal length
    trans_lists = np.array(trans_lists)
    cut_lists = [
        list(i[:seq_len].astype("str"))
        if len(i) >= seq_len
        else list(i) + list(np.zeros(seq_len - len(i)).astype("str"))
        for i in trans_lists
    ]
    cut_lists = ["".join(np.array(i).astype("str")) for i in cut_lists]

    # create a distance matrix (THIS COULD BE PARALLELIZED)
    dist = np.zeros((len(cut_lists), len(cut_lists)))

    items = [(i, j) for i in range(1, len(cut_lists)) for j in range(0, i)]
    distances = Parallel(n_jobs=n_jobs)(
        delayed(edit_distance)(cut_lists[i], cut_lists[j])
        for i, j in tqdm(items, leave=False)
    )
    for distance, (i, j) in zip(distances, items):
        dist[i, j] = distance
        dist[j, i] = distance

    # hierarchical clustering
    dists = squareform(dist)
    linkage_matrix = linkage(dists, "single")

    # make plot
    if ax == None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 10], wspace=0, hspace=0)
        ax0 = plt.subplot(gs[0])
        ax = plt.subplot(gs[1])

        dn = dendrogram(
            linkage_matrix,
            p=6,
            truncate_mode="none",
            get_leaves=True,
            orientation="left",
            no_labels=True,
            link_color_func=lambda k: "k",
            ax=ax0,
            show_contracted=False,
        )
        ax0.axis("off")
    else:
        dn = dendrogram(
            linkage_matrix,
            p=6,
            truncate_mode="none",
            get_leaves=True,
            orientation="left",
            no_labels=True,
            link_color_func=lambda k: "k",
            # ax=ax0,
            show_contracted=False,
            no_plot=True,
        )

    ax.imshow(
        color_item.swapaxes(0, 1)[np.array(dn["leaves"])],
        aspect="auto",
        interpolation=None,
        origin="lower",
    )
    ax.axis("off")

    return ax
