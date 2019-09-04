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

def subset_syllables(json_dict, indv, unit="syllables", hparams=None, include_labels = True):
    """ Grab syllables from wav data
    """
    if type(json_dict) != OrderedDict:
        json_dict = read_json(json_dict)
    # get unit info
    start_times = json_dict["indvs"][indv][unit]["start_times"]
    end_times = json_dict["indvs"][indv][unit]["stop_times"]
    if include_labels:
        labels  = json_dict["indvs"][indv][unit]["labels"]
    else:
        labels = None
    # get rate and date
    rate, data = load_wav(json_dict["wav_loc"])
    # bandpass filter
    if hparams is not None:
        data = butter_bandpass_filter(
            data, hparams.butter_lowcut, hparams.butter_highcut, rate, order=5
        )

    syllables = [
        data[int(st * rate) : int(et * rate)] for st, et in zip(start_times, end_times)
    ]
    return syllables, rate, labels


def make_spec(
    syll_wav,
    fs,
    hparams,
    mel_matrix=None,
    use_tensorflow=False,
    use_mel=True,
    return_tensor=False,
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

    return spec


def log_resize_spec(spec, scaling_factor=10):
    resize_shape = [int(np.log(np.shape(spec)[1]) * scaling_factor), np.shape(spec)[0]]
    resize_spec = Image.fromarray(spec).resize(resize_shape, Image.ANTIALIAS)
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

def create_syllable_df(dataset, indv, log_scaling_factor=10, verbosity=0, log_scale_time = True, pad_syllables=True):
    """ from a DataSet object, get all of the syllables from an individual as a spectrogram
    """
    with tqdm(total=4) as pbar:
        # get waveform of syllables
        pbar.set_description("getting syllables")
        with Parallel(n_jobs=-1, verbose=verbosity) as parallel:
            syllables = parallel(
                delayed(subset_syllables)(json_file, indv=indv, hparams=dataset.hparams)
                for json_file in tqdm(np.array(dataset.json_files)[dataset.json_indv == indv], desc="getting syllable wavs", leave=False)
            )

            # repeat rate for each wav
            syllables_sequence_id = np.concatenate([np.repeat(ii, len(i[0])) for ii, i in enumerate(syllables)])
            syllables_sequence_pos = np.concatenate([np.arange(len(i[0])) for ii, i in enumerate(syllables)])

            # list syllables
            syllables_wav = np.concatenate([i[0] for i in syllables])

            # repeat rate for each wav
            syllables_rate = np.concatenate([np.repeat(i[1], len(i[0])) for i in syllables])

            # list syllable labels
            syllables_labels = np.concatenate([i[2] for i in syllables])
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
                for syllable, rate in tqdm(zip(syllables_wav, syllables_rate), total=len(syllables_rate), desc="getting syllable spectrograms", leave=False)
            )
            pbar.update(1)
            pbar.set_description("rescaling syllables")
            # log resize spectrograms
            if log_scale_time:
                syllables_spec = parallel(
                    delayed(log_resize_spec)(
                        spec, scaling_factor = log_scaling_factor
                    )
                    for spec in tqdm(syllables_spec, desc="scaling spectrograms", leave=False)
                )
            pbar.update(1)
            pbar.set_description("padding syllables")
            # determine padding
            syll_lens = [np.shape(i)[1] for i in syllables_spec]
            pad_length = np.max(syll_lens)

            # pad syllables
            if pad_syllables:
                syllables_spec = parallel(
                    delayed(pad_spectrogram)(
                        spec, pad_length
                    )
                    for spec in tqdm(syllables_spec, desc="padding spectrograms", leave=False)
                )
            pbar.update(1)

            syllable_df = pd.DataFrame({
                'syllables_sequence_id': syllables_sequence_id,
                'syllables_sequence_pos': syllables_sequence_pos,
                'syllables_wav': syllables_wav,
                'syllables_rate': syllables_rate,
                'syllables_labels': syllables_labels,
                'syllables_spec': syllables_spec,
             })

        return syllable_df