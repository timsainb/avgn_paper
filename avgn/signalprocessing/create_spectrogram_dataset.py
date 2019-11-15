from collections import OrderedDict
from avgn.utils.json import read_json
from avgn.utils.audio import load_wav
from PIL import Image
from avgn.utils.audio import float32_to_int16, int16_to_float32
import numpy as np
from avgn.signalprocessing.spectrogramming import spectrogram
from avgn.signalprocessing.filtering import butter_bandpass_filter
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
import pandas as pd

import noisereduce as nr


def flatten_spectrograms(specs):
    return np.reshape(specs, (np.shape(specs)[0], np.prod(np.shape(specs)[1:])))


def subset_syllables(
    json_dict, indv, unit="syllables", hparams=None, include_labels=True
):
    """ Grab syllables from wav data
    """
    if type(indv) == list:
        indv = indv[0]
    if type(json_dict) != OrderedDict:
        json_dict = read_json(json_dict)
    # get unit info
    start_times = json_dict["indvs"][indv][unit]["start_times"]
    # stop times vs end_times is a quick fix that should be fixed on the parsing side
    if "end_times" in json_dict["indvs"][indv][unit].keys():
        end_times = json_dict["indvs"][indv][unit]["end_times"]
    else:
        end_times = json_dict["indvs"][indv][unit]["stop_times"]
    if include_labels:
        labels = json_dict["indvs"][indv][unit]["labels"]
    else:
        labels = None
    # get rate and date
    rate, data = load_wav(json_dict["wav_loc"])

    # convert data if needed
    if np.issubdtype(type(data[0]), np.integer):
        data = int16_to_float32(data)
    # bandpass filter
    if hparams is not None:
        data = butter_bandpass_filter(
            data, hparams.butter_lowcut, hparams.butter_highcut, rate, order=5
        )

        # reduce noise
        if hparams.reduce_noise:
            data = nr.reduce_noise(
                audio_clip=data, noise_clip=data, **hparams.noise_reduce_kwargs
            )
    syllables = [
        data[int(st * rate) : int(et * rate)] for st, et in zip(start_times, end_times)
    ]
    return syllables, rate, labels


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def make_spec(
    syll_wav,
    fs,
    hparams,
    mel_matrix=None,
    use_tensorflow=False,
    use_mel=True,
    return_tensor=False,
    norm_uint8=False,
):
    """
    """
    if use_tensorflow:
        import tensorflow as tf
        from avgn.signalprocessing.spectrogramming_tf import spectrogram_tensorflow
    # convert to float
    if type(syll_wav[0]) == int:
        syll_wav = int16_to_float32(syll_wav)

    # create spec

    if use_tensorflow:
        spec = spectrogram_tensorflow(syll_wav, fs, hparams)
        if use_mel:
            spec = tf.transpose(tf.tensordot(spec, mel_matrix, 1))
            if not return_tensor:
                spec = spec.numpy()
    else:
        spec = spectrogram(syll_wav, fs, hparams)
        if use_mel:
            spec = np.dot(spec.T, mel_matrix).T
    if norm_uint8:
        spec = (norm(spec) * 255).astype("uint8")

    return spec


def log_resize_spec(spec, scaling_factor=10):
    resize_shape = [int(np.log(np.shape(spec)[1]) * scaling_factor), np.shape(spec)[0]]
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape, Image.ANTIALIAS))
    return resize_spec


def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(
        spectrogram, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0
    )


import collections


def list_match(_list, list_of_lists):
    # Using Counter
    return [
        collections.Counter(elem) == collections.Counter(_list)
        for elem in list_of_lists
    ]


def mask_spec(spec, spec_thresh=0.9, offset=1e-10):
    """ mask threshold a spectrogram to be above some % of the maximum power
    """
    mask = spec >= (spec.max(axis=0, keepdims=1) * spec_thresh + offset)
    return spec * mask


def create_syllable_df(
    dataset,
    indv,
    unit="syllables",
    log_scaling_factor=10,
    verbosity=0,
    log_scale_time=True,
    pad_syllables=True,
    n_jobs=-1,
    include_labels=False,
):
    """ from a DataSet object, get all of the syllables from an individual as a spectrogram
    """
    with tqdm(total=4) as pbar:
        # get waveform of syllables
        pbar.set_description("getting syllables")
        with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
            syllables = parallel(
                delayed(subset_syllables)(
                    json_file,
                    indv=indv,
                    unit=unit,
                    hparams=dataset.hparams,
                    include_labels=include_labels,
                )
                for json_file in tqdm(
                    np.array(dataset.json_files)[list_match(indv, dataset.json_indv)],
                    desc="getting syllable wavs",
                    leave=False,
                )
            )

            # repeat rate for each wav
            syllables_sequence_id = np.concatenate(
                [np.repeat(ii, len(i[0])) for ii, i in enumerate(syllables)]
            )
            syllables_sequence_pos = np.concatenate(
                [np.arange(len(i[0])) for ii, i in enumerate(syllables)]
            )

            # list syllables waveforms
            syllables_wav = [
                item for sublist in [i[0] for i in syllables] for item in sublist
            ]

            # repeat rate for each wav
            syllables_rate = np.concatenate(
                [np.repeat(i[1], len(i[0])) for i in syllables]
            )

            # list syllable labels
            if syllables[0][2] is not None:
                syllables_labels = np.concatenate([i[2] for i in syllables])
            else:
                syllables_labels = None
            pbar.update(1)
            pbar.set_description("creating spectrograms")

            # create spectrograms
            syllables_spec = parallel(
                delayed(make_spec)(
                    syllable,
                    rate,
                    hparams=dataset.hparams,
                    mel_matrix=dataset.mel_matrix,
                    use_mel=True,
                    use_tensorflow=False,
                )
                for syllable, rate in tqdm(
                    zip(syllables_wav, syllables_rate),
                    total=len(syllables_rate),
                    desc="getting syllable spectrograms",
                    leave=False,
                )
            )

            # Mask spectrograms
            if dataset.hparams.mask_spec:
                syllables_spec = parallel(
                    delayed(mask_spec)(syllable, **dataset.hparams.mask_spec_kwargs)
                    for syllable in tqdm(
                        syllables_spec,
                        total=len(syllables_rate),
                        desc="masking spectrograms",
                        leave=False,
                    )
                )

            pbar.update(1)
            pbar.set_description("rescaling syllables")
            # log resize spectrograms
            if log_scale_time:
                syllables_spec = parallel(
                    delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
                    for spec in tqdm(
                        syllables_spec, desc="scaling spectrograms", leave=False
                    )
                )
            pbar.update(1)
            pbar.set_description("padding syllables")
            # determine padding
            syll_lens = [np.shape(i)[1] for i in syllables_spec]
            pad_length = np.max(syll_lens)

            # pad syllables
            if pad_syllables:
                syllables_spec = parallel(
                    delayed(pad_spectrogram)(spec, pad_length)
                    for spec in tqdm(
                        syllables_spec, desc="padding spectrograms", leave=False
                    )
                )
            pbar.update(1)

            syllable_df = pd.DataFrame(
                {
                    "syllables_sequence_id": syllables_sequence_id,
                    "syllables_sequence_pos": syllables_sequence_pos,
                    "syllables_wav": syllables_wav,
                    "syllables_rate": syllables_rate,
                    "syllables_labels": syllables_labels,
                    "syllables_spec": syllables_spec,
                }
            )

        return syllable_df


def get_element(
    datafile, indv=None, element_number=1, element="syllable", hparams=None
):

    # if an individual isnt specified, grab the first one
    if indv == None:
        indv = datafile.indvs[0]

    # get the element
    element = datafile.data["indvs"][indv][element]

    # get the part of the wav we want to load
    st = element["start_times"][element_number]
    et = element["end_times"][element_number]

    # load the data
    rate, element = load_wav(
        datafile.data["wav_loc"], offset=st, duration=et - st, sr=None
    )

    if np.issubdtype(type(element[0]), np.integer):
        element = int16_to_float32(data)

    if hparams is not None:
        element = butter_bandpass_filter(
            element, hparams.butter_lowcut, hparams.butter_highcut, rate, order=5
        )

    return rate, element


def prepare_wav(wav_loc, hparams=None):
    """ load wav and convert to correct format
    """

    # get rate and date
    rate, data = load_wav(wav_loc)

    # convert data if needed
    if np.issubdtype(type(data[0]), np.integer):
        data = int16_to_float32(data)
    # bandpass filter
    if hparams is not None:
        data = butter_bandpass_filter(
            data, hparams.butter_lowcut, hparams.butter_highcut, rate, order=5
        )

        # reduce noise
        if hparams.reduce_noise:
            data = nr.reduce_noise(
                audio_clip=data, noise_clip=data, **hparams.noise_reduce_kwargs
            )

    return rate, data


def create_label_df(
    json_dict,
    hparams=None,
    labels_to_retain=[],
    unit="syllables",
    dict_features_to_retain=[],
    key=None,
):
    """ create a dataframe from json dictionary of time events and labels
    """

    syllable_dfs = []
    # loop through individuals
    for indvi, indv in enumerate(json_dict["indvs"].keys()):
        if unit not in json_dict["indvs"][indv].keys():
            continue
        indv_dict = {}
        indv_dict["start_time"] = json_dict["indvs"][indv][unit]["start_times"]
        indv_dict["end_time"] = json_dict["indvs"][indv][unit]["end_times"]

        # get data for individual
        for label in labels_to_retain:
            indv_dict[label] = json_dict["indvs"][indv][unit][label]
            if len(indv_dict[label]) < len(indv_dict["start_time"]):
                indv_dict[label] = np.repeat(
                    indv_dict[label], len(indv_dict["start_time"])
                )

        # create dataframe
        indv_df = pd.DataFrame(indv_dict)
        indv_df["indv"] = indv
        indv_df["indvi"] = indvi
        syllable_dfs.append(indv_df)

    syllable_df = pd.concat(syllable_dfs)
    for feat in dict_features_to_retain:
        syllable_df[feat] = json_dict[feat]
    # associate current syllables with key
    syllable_df["key"] = key

    return syllable_df


def get_row_audio(syllable_df, wav_loc, hparams):
    """ load audio and grab individual syllables
    TODO: for large sparse WAV files, the audio should be loaded only for the syllable
    """

    # load audio
    rate, data = prepare_wav(wav_loc, hparams)
    data = data.astype('float32')
    
    # get audio for each syllable
    syllable_df["audio"] = [
        data[int(st * rate) : int(et * rate)]
        for st, et in zip(syllable_df.start_time.values, syllable_df.end_time.values)
    ]

    syllable_df["rate"] = rate

    return syllable_df
