from scipy.signal import butter, lfilter
import numpy as np
#from librosa.core.time_frequency import mel_frequencies
from librosa import mel_frequencies
import warnings


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    if highcut > int(fs / 2):
        warnings.warn("Highcut is too high for bandpass filter. Setting to nyquist")
        highcut = int(fs / 2)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def window_rms(a, window_size):
    a2 = np.power(a, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(a2, window, "valid"))


def RMS(data, rate, rms_stride, rms_window, rms_padding, noise_thresh):
    """
    Take data, run and RMS filter over it
    """

    # we compute root mean squared over a window, where we stride by rms_stride seconds for speed
    rms_data = window_rms(
        data.astype("float32")[:: int(rms_stride * rate)],
        int(rate * rms_window * rms_stride),
    )
    rms_data = rms_data / np.max(rms_data)

    # convolve a block filter over RMS, then threshold it, so to call everything with RMS > noise_threshold noise
    block_filter = np.ones(int(rms_padding * rms_stride * rate))  # create our filter

    # pad the data to be filtered
    rms_threshed = np.concatenate(
        (
            np.zeros(int(len(block_filter) / 2)),
            np.array(rms_data > noise_thresh),
            np.zeros(int(len(block_filter) / 2)),
        )
    )
    # convolve on our filter
    sound_threshed = np.array(np.convolve(rms_threshed, block_filter, "valid") > 0)[
        : len(rms_data)
    ]

    return rms_data, sound_threshed


import sys
import os


def prepare_mel_matrix(hparams, rate, return_numpy=True, GPU_backend=False):
    """ Create mel filter
    """
    # import tensorflow if needed
    if "tf" not in sys.modules:
        if not GPU_backend:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import tensorflow as tf

    # create a filter to convolve with the spectrogram
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=hparams.num_mel_bins,
        num_spectrogram_bins=int(hparams.n_fft / 2) + 1,
        sample_rate=rate,
        lower_edge_hertz=hparams.mel_lower_edge_hertz,
        upper_edge_hertz=hparams.mel_upper_edge_hertz,
        dtype=tf.dtypes.float32,
        name=None,
    )

    # gets the center frequencies of mel bands
    mel_f = mel_frequencies(
        n_mels=hparams.num_mel_bins + 2,
        fmin=hparams.mel_lower_edge_hertz,
        fmax=hparams.mel_upper_edge_hertz,
    )

    # Slaney-style mel is scaled to be approx constant energy per channel (from librosa)
    enorm = tf.dtypes.cast(
        tf.expand_dims(
            tf.constant(
                2.0
                / (mel_f[2 : hparams.num_mel_bins + 2] - mel_f[: hparams.num_mel_bins])
            ),
            0,
        ),
        tf.float32,
    )

    mel_matrix = tf.multiply(mel_matrix, enorm)
    mel_matrix = tf.divide(mel_matrix, tf.reduce_sum(mel_matrix, axis=0))
    if return_numpy:
        return mel_matrix.numpy()
    else:
        return mel_matrix
