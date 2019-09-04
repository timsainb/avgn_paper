# any snippits of code that don't fit elsewhere

import numpy as np
import pickle

import zipfile
from avgn.utils.paths import ensure_dir
from tqdm.autonotebook import tqdm


def prepare_env(GPU=[]):
    import IPython

    ipython = IPython.get_ipython()
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    ipython.magic("env CUDA_VISIBLE_DEVICES=GPU")


def zero_one_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def save_dict_pickle(dict_, save_loc):
    with open(save_loc, "wb") as f:
        pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)


def rescale(X, out_min, out_max):
    return out_min + (X - np.min(X)) * ((out_max - out_min) / (np.max(X) - np.min(X)))


def seconds_to_str(seconds):
    """ converts a number of seconds to hours, minutes, seconds.ms
    """
    (hours, remainder) = divmod(seconds, 3600)
    (minutes, seconds) = divmod(remainder, 60)
    return "h{}m{}s{}".format(int(hours), int(minutes), float(seconds))


def unzip_file(zip_path, directory_to_extract_to):
    """ unzip file using tqdm
    """
    ensure_dir(directory_to_extract_to)
    with zipfile.ZipFile(file=zip_path) as zip_file:
        # Loop over each file
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            zip_file.extract(member=file, path=directory_to_extract_to)

