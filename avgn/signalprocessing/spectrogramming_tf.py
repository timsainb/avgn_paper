# https://github.com/keithito/tacotron/blob/master/util/audio.py
import tensorflow as tf
import numpy as np
import librosa


def _normalize_tensorflow(S, hparams):
    return tf.clip_by_value((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def _amp_to_db_tensorflow(x):
    return 20 * _tf_log10(tf.clip_by_value(tf.abs(x), 1e-5, 1e100))


def _stft_tensorflow(signals, fs, hparams):
    win_length = hparams.win_length_ms / 1000 * fs
    hop_length = hparams.hop_length_ms / 1000 * fs
    return tf.signal.stft(
        signals,
        win_length,
        hop_length,
        hparams.n_fft,
        pad_end=True,
        window_fn=tf.signal.hann_window,
    )


def spectrogram_tensorflow(y, fs, hparams):
    D = _stft_tensorflow(y, fs, hparams)
    S = _amp_to_db_tensorflow(tf.abs(D)) - hparams.ref_level_db
    return _normalize_tensorflow(S, hparams)


def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


# use this one when istft is fixed!
def _istft_tensorflow(stfts, fs, hparams):
    win_length = hparams.win_length_ms / 1000 * fs
    hop_length = hparams.hop_length_ms / 1000 * fs
    return tf.signal.inverse_stft(stfts, win_length, hop_length, hparams.n_fft)


def _denormalize_tensorflow(S, hparams):
    return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


from avgn.signalprocessing.spectrogramming import _stft, _istft


def _griffin_lim_tensorflow(S, fs, hparams, use_tf_istft=False):
    """TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb and
  https://github.com/keithito/tacotron/blob/master/util/audio.py
  issue: https://github.com/tensorflow/tensorflow/issues/28444
  """
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    if use_tf_istft:
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = tf.py_function(_istft(S_complex, fs, hparams))
        for _ in range(hparams.griffin_lim_iters):
            est = _stft_tensorflow(y, fs, hparams)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles, fs, hparams)
        return tf.squeeze(y, 0)

    else:
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = _istft(S_complex * angles, fs, hparams)
        for _ in range(hparams.griffin_lim_iters):
            angles = np.exp(1j * np.angle(_stft(y, fs, hparams)))
            y = _istft(S_complex * angles, fs, hparams)
        return y


def inv_spectrogram_tensorflow(spectrogram, fs, hparams):
    """Converts spectrogram to waveform using librosa"""
    S = _db_to_amp_tensorflow(
        _denormalize_tensorflow(spectrogram, hparams) + hparams.ref_level_db
    )  # Convert back to linear
    return _griffin_lim_tensorflow(S ** hparams.power, fs, hparams)  # Reconstruct phase
