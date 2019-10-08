import librosa
import librosa.filters
import numpy as np
from scipy import signal


def spectrogram_nn(y, fs, hparams):
    D = _stft(preemphasis(y, hparams), fs, hparams)
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return S


def melspectrogram_nn(y, fs, hparams, _mel_basis):
    D = _stft(preemphasis(y, hparams), fs, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), _mel_basis)) - hparams.ref_level_db
    return S


def melspectrogram(y, fs, hparams, _mel_basis):
    return _normalize(melspectrogram_nn(y, fs, hparams, _mel_basis), hparams)


def spectrogram(y, fs, hparams):
    return _normalize(spectrogram_nn(y, fs, hparams), hparams)


def reassigned_spectrogram(y, fs, hparams):

    freqs, times, mags = librosa.reassigned_spectrogram(
        y=preemphasis(y, hparams),
        sr=fs,
        n_fft=hparams.n_fft,
        hop_length=int(hparams.hop_length_ms / 1000 * fs),
        win_length=int(hparams.win_length_ms / 1000 * fs),
        center=False,
    )
    S = librosa.amplitude_to_db(
        (freqs > 0) * (times > 0) * mags, ref=hparams.ref_level_db
    )

    S = _normalize(S, hparams)
    return S
    # return freqs, times, mags


def inv_spectrogram(spectrogram, fs, hparams):
    """Converts spectrogram to waveform using librosa"""
    S = _db_to_amp(
        _denormalize(spectrogram, hparams) + hparams.ref_level_db
    )  # Convert back to linear
    return inv_preemphasis(
        _griffin_lim(S ** hparams.power, fs, hparams), hparams
    )  # Reconstruct phase


def preemphasis(x, hparams):
    return signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x, hparams):
    return signal.lfilter([1], [1, -hparams.preemphasis], x)


def _griffin_lim(S, fs, hparams):
    """librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, fs, hparams)
    for _ in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, fs, hparams)))
        y = _istft(S_complex * angles, fs, hparams)
    return y


def _stft(y, fs, hparams):
    return librosa.stft(
        y=y,
        n_fft=hparams.n_fft,
        hop_length=int(hparams.hop_length_ms / 1000 * fs),
        win_length=int(hparams.win_length_ms / 1000 * fs),
    )


def _istft(y, fs, hparams):
    hop_length = int(hparams.hop_length_ms / 1000 * fs)
    win_length = int(hparams.win_length_ms / 1000 * fs)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _linear_to_mel(spectrogram, _mel_basis):
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis(hparams):
    n_fft = (hparams.num_freq - 1) * 2
    return librosa.filters.mel(
        hparams.sample_rate,
        n_fft,
        n_mels=hparams.num_mels,
        fmin=hparams.fmin_mel,
        fmax=hparams.fmax_mel,
    )


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S, hparams):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S, hparams):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

