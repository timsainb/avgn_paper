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


def spectrogram_librosa(y, fs, hparams):
    D = np.abs(_stft(preemphasis(y, hparams), fs, hparams))
    S = librosa.amplitude_to_db(D) - hparams.ref_level_db
    S_norm = _normalize(S, hparams)
    return S_norm


def melspectrogram_librosa(y, fs, hparams, _mel_basis):
    D = _stft(preemphasis(y, hparams), fs, hparams)
    S = (
        librosa.amplitude_to_db(_linear_to_mel(np.abs(D), _mel_basis))
        - hparams.ref_level_db
    )
    return S


def griffinlim_librosa(spectrogram, fs, hparams):
    hop_length = int(hparams.hop_length_ms / 1000 * fs)
    win_length = int(hparams.win_length_ms / 1000 * fs)
    return inv_preemphasis(
        librosa.griffinlim(
            spectrogram,
            n_iter=hparams.griffin_lim_iters,
            hop_length=hop_length,
            win_length=win_length,
        ),
        hparams,
    )


def inv_spectrogram_librosa(spectrogram, fs, hparams):
    """Converts spectrogram to waveform using librosa"""
    S_denorm = _denormalize(spectrogram, hparams)
    S = librosa.db_to_amplitude(
        S_denorm + hparams.ref_level_db
    )  # Convert back to linear
    # Reconstruct phase
    return griffinlim_librosa(S, fs, hparams)


def inv_spectrogram(spectrogram, fs, hparams):
    """Converts spectrogram to waveform using librosa"""
    S = _db_to_amp(
        _denormalize(spectrogram, hparams) + hparams.ref_level_db
    )  # Convert back to linear
    return inv_preemphasis(
        _griffin_lim(S ** hparams.power, fs, hparams), hparams
    )  # Reconstruct phase


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


def _mel_to_linear(melspectrogram, _mel_basis=None, _mel_inverse_basis=None):
    if (_mel_basis is None) and (_mel_inverse_basis is None):
        raise ValueError("_mel_basis or _mel_inverse_basis needed")
    elif _mel_inverse_basis is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            _mel_inverse_basis = np.nan_to_num(
                np.divide(_mel_basis, np.sum(_mel_basis.T, axis=1))
            ).T
    return np.matmul(_mel_inverse_basis, melspectrogram)


def _build_mel_inversion_basis(_mel_basis):
    with np.errstate(divide="ignore", invalid="ignore"):
        mel_inverse_basis = np.nan_to_num(
            np.divide(_mel_basis, np.sum(_mel_basis.T, axis=1))
        ).T
    return mel_inverse_basis


def _build_mel_basis(hparams, fs, rate=None, use_n_fft=True):
    if "n_fft" not in hparams.__dict__ or (use_n_fft == False):
        if "num_freq" in hparams.__dict__:
            n_fft = (hparams.num_freq - 1) * 2
        else:
            n_fft = int(hparams.win_length_ms / 1000 * fs)
    else:
        n_fft = hparams.n_fft
    if rate is None:
        rate = hparams.sample_rate
    _mel_basis = librosa.filters.mel(
        rate,
        n_fft,
        n_mels=hparams.num_mels,
        fmin=hparams.mel_lower_edge_hertz,
        fmax=hparams.mel_upper_edge_hertz,
    )

    return np.nan_to_num(_mel_basis.T / np.sum(_mel_basis, axis=1)).T


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S, hparams):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S, hparams):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

