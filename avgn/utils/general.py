# any snippits of code that don't fit elsewhere

import numpy as np
import pickle 

def prepare_env(GPU=[]):
    import IPython 
    ipython = IPython.get_ipython()
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    ipython.magic("env CUDA_VISIBLE_DEVICES=GPU")

def zero_one_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def save_dict_pickle(dict_, save_loc):
    with open(save_loc, 'wb') as f:
        pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)

def rescale(X, out_min, out_max):
    return out_min + (X - np.min(X)) * ((out_max - out_min) / (np.max(X) - np.min(X)))

class HParams(object):
    """ Hparams was removed from tf 2.0alpha so this is a placeholder
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


def seconds_to_str(seconds):
    """ converts a number of seconds to hours, minutes, seconds.ms
    """
    (hours, remainder) = divmod(seconds, 3600)
    (minutes, seconds) = divmod(remainder, 60)
    return "h{}m{}s{}".format(int(hours), int(minutes), float(seconds))