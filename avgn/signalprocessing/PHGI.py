# from https://github.com/andimarafioti/tifresi/
from __future__ import print_function, division


from ltfatpy import dgtreal, idgtreal
from ltfatpy.gabor.gabdual import gabdual
import numpy as np
import librosa
from avgn.signalprocessing.spectrogramming import (
    _linear_to_mel,
    _normalize,
    _amp_to_db,
    _db_to_amp,
    _denormalize,
)

from ltfatpy import dgtreal, idgtreal


def _analysis_window(x, hop_size, n_fft):
    """
    Window for gaussian stft
    """
    return {"name": "gauss", "tfr": hop_size * n_fft / len(x)}


def dgt(x, hop_size, n_fft):
    """Compute the DGT of a real signal with a gauss window."""
    assert len(x.shape) == 1
    assert np.mod(len(x), hop_size) == 0
    assert np.mod(n_fft, 2) == 0, "The number of stft channels needs to be even"
    assert np.mod(len(x), n_fft) == 0
    g_analysis = _analysis_window(x, hop_size, n_fft)
    return dgtreal(x.astype(np.float64), g_analysis, hop_size, n_fft)[0]


# This function might need another name
def preprocess_signal(y, M=1024):
    """Trim and cut signal.
    
    The function ensures that the signal length is a multiple of M.
    """
    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    # y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Padding
    left_over = np.mod(len(y), M)
    extra = M - left_over
    y = np.pad(y, (0, extra))
    assert np.mod(len(y), M) == 0
    return y


def spectrogram_no_norm(y, fs, hparams):
    # preprocess to make signal correct length
    hop_size = int(hparams.hop_length_ms / 1000 * fs)
    n_fft = int(hparams.win_length_ms / 1000 * fs)
    y = preprocess_signal(y, M=n_fft)
    # stft
    """Compute the spectrogram of a real signal."""
    D = np.abs(dgt(y, hop_size=hop_size, n_fft=n_fft))
    return D


def spectrogram(y, fs, hparams):
    # preprocess to make signal correct length
    hop_size = int(hparams.hop_length_ms / 1000 * fs)
    n_fft = int(hparams.win_length_ms / 1000 * fs)
    y = preprocess_signal(y, M=n_fft)
    # stft
    """Compute the spectrogram of a real signal."""
    D = np.abs(dgt(y, hop_size=hop_size, n_fft=n_fft))
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S, hparams)


def melspectrogram(y, fs, hparams, _mel_basis):
    S = spectrogram(y, fs, hparams)
    return _linear_to_mel(S, _mel_basis)


def _synthesis_window(X, hop_size, n_fft):
    """
    window spectrogram inversion
    """
    L = hop_size * X.shape[1]
    tfr = hop_size * n_fft / L
    g_analysis = {"name": "gauss", "tfr": tfr}
    return {"name": ("dual", g_analysis["name"]), "tfr": tfr}


def idgt(X, hop_size, n_fft):
    """Compute the inverse DGT of real signal x with a gauss window."""
    assert len(X.shape) == 2
    assert np.mod(n_fft, 2) == 0, "The number of stft channels needs to be even"
    assert X.shape[0] == n_fft // 2 + 1
    g_synthesis = _synthesis_window(X, hop_size, n_fft)
    return idgtreal(X.astype(np.complex128), g_synthesis, hop_size, n_fft)[0]


def invert_spectrogram(spectrogram, fs, hparams):
    spectrogram = _db_to_amp(
        _denormalize(spectrogram, hparams) + hparams.ref_level_db
    )  # Convert back to linear

    """Invert a spectrogram by reconstructing the phase with PGHI."""
    hop_size = int(hparams.hop_length_ms / 1000 * fs)
    n_fft = int(hparams.win_length_ms / 1000 * fs)

    audio_length = hop_size * spectrogram.shape[1]
    tfr = hop_size * n_fft / audio_length
    g_analysis = {"name": "gauss", "tfr": tfr}

    tgrad, fgrad = modgabphasegrad("abs", spectrogram, g_analysis, hop_size, n_fft)
    phase = pghi(spectrogram, tgrad, fgrad, hop_size, n_fft, audio_length)

    reComplexStft = spectrogram * np.exp(1.0j * phase)

    return idgt(reComplexStft, hop_size, n_fft)


# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
# Credits
# #######
#
# Copyright(c) 2015-2018
# ----------------------
#
# * `LabEx Archimède <http://labex-archimede.univ-amu.fr/>`_
# * `Laboratoire d'Informatique Fondamentale <http://www.lif.univ-mrs.fr/>`_
#   (now `Laboratoire d'Informatique et Systèmes <http://www.lis-lab.fr/>`_)
# * `Institut de Mathématiques de Marseille <http://www.i2m.univ-amu.fr/>`_
# * `Université d'Aix-Marseille <http://www.univ-amu.fr/>`_
#
# This software is a port from LTFAT 2.1.0 :
# Copyright (C) 2005-2018 Peter L. Soendergaard <peter@sonderport.dk>.
#
# Contributors
# ------------
#
# * Denis Arrivault <contact.dev_AT_lis-lab.fr>
# * Florent Jaillet <contact.dev_AT_lis-lab.fr>
#
# Description
# -----------
#
# ltfatpy is a partial Python port of the
# `Large Time/Frequency Analysis Toolbox <http://ltfat.sourceforge.net/>`_,
# a MATLAB®/Octave toolbox for working with time-frequency analysis and
# synthesis.
#
# Version
# -------
#
# * ltfatpy version = 1.0.16
# * LTFAT version = 2.1.0
#
# Licence
# -------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ######### COPYRIGHT #########


"""Module of phase gradient computation
Ported from ltfat_2.1.0/gabor/gabphasegrad.m
.. moduleauthor:: Florent Jaillet
"""


import numpy as np

from ltfatpy.comp.comp_sigreshape_pre import comp_sigreshape_pre
from ltfatpy.gabor.dgtlength import dgtlength
from ltfatpy.gabor.gabwin import gabwin
from ltfatpy.tools.postpad import postpad
from ltfatpy.fourier.fftindex import fftindex
from ltfatpy.comp.comp_sepdgt import comp_sepdgt
from ltfatpy.fourier.pderiv import pderiv


def modgabphasegrad(method, *args, **kwargs):
    """Modified Phase gradient of the discrete Gabor transform
	We modified this to work with dgtreals on the phase and abs case
	Phase case we did a lot of changes,
	abs case we added M as a mandatory parameter
    - Usage:
        | ``(tgrad, fgrad, c) = gabphasegrad('dgt', f, g, a, M, L=None)``
        | ``(tgrad, fgrad) = gabphasegrad('phase', cphase, a)``
        | ``(tgrad, fgrad) = gabphasegrad('abs', s, g, a, M, difforder=2)``
    - Input parameters:
    :param str method: Method used to compute the phase gradient, see the
        possible values below
    :param numpy.ndarray f: (defined if ``method='dgt'``) Input signal
    :param numpy.ndarray cphase: (defined if ``method='phase'``) Phase of a
        :func:`~ltfatpy.gabor.dgt.dgt` of the signal
    :param numpy.ndarray s: (defined if ``method='abs'``) Spectrogram of the
        signal
    :param numpy.ndarray g: (defined if ``method='dgt'`` or ``method='phase'``)
        Window function
    :param int a: (defined if ``method='dgt'`` or ``method='phase'`` or
        ``method='abs'``) Length of time shift
    :param int M: (defined if ``method='dgt'``) Number of channels
    :param int L: (defined if ``method='dgt'``, optional) Length of transform
        to do
    :param int difforder: (defined if ``method='abs'``, optional) Order of the
        centered finite difference scheme used to perform the needed numerical
        differentiation
    - Output parameters:
    :returns: ``(tgrad, fgrad, c)`` if ``method='dgt'``, or ``(tgrad, fgrad)``
        if ``method='phase'`` or ``method='abs'``
    :rtype: tuple
    :var numpy.ndarray tgrad: Instantaneous frequency
    :var numpy.ndarray fgrad: Local group delay
    :var numpy.ndarray c: Gabor coefficients
    ``gabphasegrad`` computes the time-frequency gradient of the phase of the
    :func:`~ltfatpy.gabor.dgt.dgt` of a signal. The derivative in time
    **tgrad** is the instantaneous frequency while the frequency derivative
    **fgrad** is the local group delay.
    **tgrad** and **fgrad** measure the deviation from the current time and
    frequency, so a value of zero means that the instantaneous frequency is
    equal to the center frequency of the considered channel.
    **tgrad** is scaled such that distances are measured in samples. Similarly,
    **fgrad** is scaled such that the Nyquist frequency (the highest possible
    frequency) corresponds to a value of ``L/2``.
    The computation of **tgrad** and **fgrad** is inaccurate when the absolute
    value of the Gabor coefficients is low. This is due to the fact the the
    phase of complex numbers close to the machine precision is almost
    random. Therefore, **tgrad** and **fgrad** may attain very large random
    values when ``abs(c)`` is close to zero.
    The computation can be done using three different methods:
        =========== ===========================================================
        ``'dgt'``   Directly from the signal.
        ``'phase'`` From the phase of a :func:`~ltfatpy.gabor.dgt.dgt` of the
                    signal. This is the classic method used in the phase
                    vocoder.
        ``'abs'``   From the absolute value of the
                    :func:`~ltfatpy.gabor.dgt.dgt`. Currently this method works
                    only for Gaussian windows.
        =========== ===========================================================
    ``(tgrad, fgrad, c) = gabphasegrad('dgt', f, g, a, M)`` computes the
    time-frequency gradient using a :func:`~ltfatpy.gabor.dgt.dgt` of the
    signal **f**. The :func:`~ltfatpy.gabor.dgt.dgt` is computed using the
    window **g** on the lattice specified by the time shift **a** and the
    number of channels **M**. The algorithm used to perform this calculation
    computes several DGTs, and therefore this routine takes the exact same
    input parameters as :func:`~ltfatpy.gabor.dgt.dgt`.
    The window **g** may be specified as in :func:`~ltfatpy.gabor.dgt.dgt`. If
    the window used is ``'gauss'``, the computation will be done by a faster
    algorithm.
    ``(tgrad, fgrad, c) = gabphasegrad('dgt', f, g, a, M)`` additionally
    returns the Gabor coefficients ``c``, as they are always computed as a
    byproduct of the algorithm.
    ``(tgrad, fgrad) = gabphasegrad('phase', cphase, a)`` computes the phase
    gradient from the phase **cphase** of a :func:`~ltfatpy.gabor.dgt.dgt` of
    the signal. The original :func:`~ltfatpy.gabor.dgt.dgt` from which the
    phase is obtained must have been computed using a time-shift of **a**.
    ``(tgrad, fgrad) = gabphasegrad('abs', s, g, a)`` computes the phase
    gradient from the spectrogram **s**. The spectrogram must have been
    computed using the window **g** and time-shift **a**.
    ``(tgrad, fgrad) = gabphasegrad('abs', s, g, a, difforder=ord)`` uses a
    centered finite difference scheme of order ``ord`` to perform the needed
    numerical differentiation. Default is to use a 4th order scheme.
    Currently the 'abs' method only works if the window **g** is a Gaussian
    window specified as a string or cell array.
    .. seealso:: :func:`resgram`, :func:`gabreassign`,
                 :func:`~ltfatpy.gabor.dgt.dgt`
    - References:
        :cite:`aufl95,cmdaaufl97,fl65`
    """

    # NOTE: This function doesn't support the parameter lt (lattice type)
    # supported by the corresponding octave function and the lattice used is
    # seperable (square lattice lt = (0, 1)).

    # NOTE: As in the octave version of this function, if needed, the
    # undocumented optional keyword minlvl is available when using method=dgt.
    # So it can be passed using a call of the following form:
    # (tgrad, fgrad, c) = gabphasegrad('dgt', f, g, a, M, minlvl=val)

    if not isinstance(method, str):
        raise TypeError(
            "First argument must be a str containing the method "
            'name, "dgt", "phase" or "abs".'
        )

    method = method.lower()

    if method == "dgt":
        raise Exception("We dont know if this works")
        # ---------------------------  DGT method ------------------------

        (f, g, a, M) = args

        if "L" in kwargs:
            L = kwargs["L"]
        else:
            L = None

        if "minlvl" in kwargs:
            minlvl = kwargs["minlvl"]
        else:
            minlvl = np.finfo(np.float64).tiny

            # # ----- step 1 : Verify f and determine its length -------
            # Change f to correct shape.

        f, Ls, W, wasrow, remembershape = comp_sigreshape_pre(f, 0)

        # # ------ step 2: Verify a, M and L
        if not L:
            # ----- step 2b : Verify a, M and get L from the signal length f---
            L = dgtlength(Ls, a, M)
        else:
            # ----- step 2a : Verify a, M and get L
            Luser = dgtlength(L, a, M)
            if Luser != L:
                raise ValueError(
                    "Incorrect transform length L = {0:d} "
                    "specified. Next valid length is L = {1:d}. "
                    "See the help of dgtlength for the "
                    "requirements.".format(L, Luser)
                )

                # # ----- step 3 : Determine the window
        g, info = gabwin(g, a, M, L)

        if L < info["gl"]:
            raise ValueError("Window is too long.")

            # # ----- step 4: final cleanup ---------------

        f = postpad(f, L)

        # # ------ algorithm starts --------------------

        # Compute the time weighted version of the window.
        hg = fftindex(L) * g

        # The computation done this way is insensitive to whether the dgt is
        # phaselocked or not.
        c = comp_sepdgt(f, g, a, M, 0)

        c_h = comp_sepdgt(f, hg, a, M, 0)

        c_s = np.abs(c) ** 2

        # Remove small values because we need to divide by c_s
        c_s = np.maximum(c_s, minlvl * np.max(c_s))

        # Compute the group delay
        fgrad = np.real(c_h * c.conjugate() / c_s)

        if info["gauss"]:
            # The method used below only works for the Gaussian window, because
            # the time derivative and the time multiplicative of the Gaussian
            # are identical.
            tgrad = np.imag(c_h * c.conjugate() / c_s) / info["tfr"]

        else:
            # The code below works for any window, and not just the Gaussian

            dg = pderiv(g, difforder=float("inf")) / (2 * np.pi)
            c_d = comp_sepdgt(f, dg, a, M, 0)

            # NOTE: There is a bug here in the original octave file as it
            # contains a reshape that uses an undefined variable N.
            # You can get the error with LTFAT 2.1.0 in octave by running for
            # example:
            # gabphasegrad('dgt', rand(16,1), rand(16,1), 4, 16)
            #
            # So we just comment out the corresponding line here, as it appears
            # to be unneeded:
            # c_d.shape = (M, N, W)

            # Compute the instantaneous frequency
            tgrad = -np.imag(c_d * c.conjugate() / c_s)

        return (tgrad, fgrad, c)

    elif method == "phase":

        # ---------------------------  phase method ------------------------

        (cphase, a, M) = args

        if not np.isrealobj(cphase):
            raise TypeError(
                "Input phase must be real valued. Use the 'angle'"
                " function to compute the argument of complex "
                "numbers."
            )

            # --- linear method ---
        if cphase.ndim == 3:
            M2, N, W = cphase.shape  # M2 is the number of channels from 0 to Nyquist
        else:
            M2, N = cphase.shape  # M2 is the number of channels from 0 to Nyquist
        L = N * a
        b = L / M

        # Forward approximation
        tgrad_1 = cphase - np.roll(cphase, -1, axis=1)

        # numpy round function doesn't use the same convention than octave for
        # half-integers but the standard Python round function uses the same
        # convention than octave, so we use the Python standard round in the
        # computation below
        octave_round = np.vectorize(round)
        tgrad_1 = tgrad_1 - 2 * np.pi * octave_round(tgrad_1 / (2 * np.pi))
        # Backward approximation
        tgrad_2 = np.roll(cphase, 1, axis=1) - cphase
        tgrad_2 = tgrad_2 - 2 * np.pi * octave_round(tgrad_2 / (2 * np.pi))
        # Average
        tgrad = (tgrad_1 + tgrad_2) / 2

        tgrad = -tgrad / (2 * np.pi * a) * L

        # Phase-lock the angles.
        TimeInd = np.arange(N) * a
        FreqInd = np.arange(M2) / M

        phl = np.dot(
            FreqInd.reshape((FreqInd.shape[0], 1)),
            TimeInd.reshape((1, TimeInd.shape[0])),
        )
        # NOTE: in the following lines, the shape of phl is changed so that
        # broadcasting works in the following addition with cphase when cphase
        # has more than two dimensions
        new_shape = np.ones((len(cphase.shape),), dtype=int)
        new_shape[0] = phl.shape[0]
        new_shape[1] = phl.shape[1]
        phl = phl.reshape(tuple(new_shape))
        cphase = cphase + 2 * np.pi * phl
        cphase_to_aprox = np.concatenate([-cphase[1:2], cphase, -cphase[-2:-1]])

        # Forward approximation
        fgrad_1 = cphase_to_aprox - np.roll(cphase_to_aprox, -1, axis=0)
        fgrad_1 = fgrad_1 - 2 * np.pi * octave_round(fgrad_1 / (2 * np.pi))
        fgrad_1 = fgrad_1[1:-1]
        # Backward approximation
        fgrad_2 = np.roll(cphase_to_aprox, 1, axis=0) - cphase_to_aprox
        fgrad_2 = fgrad_2 - 2 * np.pi * octave_round(fgrad_2 / (2 * np.pi))
        fgrad_2 = fgrad_2[1:-1]
        # Average
        fgrad = (fgrad_1 + fgrad_2) / 2

        fgrad = fgrad / (2 * np.pi * b) * L

        return (tgrad, fgrad)

    elif method == "abs":
        # ---------------------------  abs method ------------------------

        (s, g, a, M) = args

        if "difforder" in kwargs:
            difforder = kwargs["difforder"]
        else:
            difforder = 2

        if not np.all(s >= 0.0):
            raise ValueError("First input argument must be positive or zero.")

        if s.ndim == 3:
            M2, N, W = s.shape
        else:
            M2, N = s.shape

        L = N * a

        g, info = gabwin(g, a, M, L)

        if not info["gauss"]:
            raise ValueError(
                "The window must be a Gaussian window (specified "
                "as a string or as a dictionary)."
            )

        b = L / M
        # We must avoid taking the log of zero.
        # Therefore we add the smallest possible
        # number
        logs = np.log(s + np.finfo(s.dtype).tiny)

        # XXX REMOVE Add a small constant to limit the dynamic range. This
        # should lessen the problem of errors in the differentiation for points
        # close to (but not exactly) zeros points.
        maxmax = np.max(logs)
        tt = -11.0
        logs[logs < (maxmax + tt)] = tt

        fgrad = pderiv(logs, 1, difforder) / (2 * np.pi) * info["tfr"]
        tgrad = pderiv(logs, 0, difforder) / (2 * np.pi * info["tfr"]) * (M / M2)
        # Fix the first and last rows .. the
        # borders are symmetric so the centered difference is 0
        tgrad[0, :] = 0
        tgrad[-1, :] = 0

        return (tgrad, fgrad)

    else:
        raise ValueError(
            "First argument must be the method name, 'dgt', " "'phase' or 'abs'."
        )


import heapq
import numba
from numba import njit

__author__ = "Andres"


@njit
def pghi(spectrogram, tgrad, fgrad, a, M, L, tol=1e-7):
    """"Implementation of "A noniterativemethod for reconstruction of phase from STFT magnitude". by Prusa, Z., Balazs, P., and Sondergaard, P. Published in IEEE/ACM Transactions on Audio, Speech and LanguageProcessing, 25(5):1154–1164 on 2017. 
    a = hop size
    M = fft window size
    L = signal length
    tol = tolerance under the max value of the spectrogram
    """
    spectrogram = spectrogram.copy()
    abstol = np.array([1e-10], dtype=spectrogram.dtype)[
        0
    ]  # if abstol is not the same type as spectrogram then casting occurs
    phase = np.zeros_like(spectrogram)
    max_val = np.amax(spectrogram)  # Find maximum value to start integration
    max_x, max_y = np.where(spectrogram == max_val)
    max_pos = max_x[0], max_y[0]

    if (
        max_val <= abstol
    ):  # Avoid integrating the phase for the spectogram of a silent signal
        print("Empty spectrogram")
        return phase

    M2 = spectrogram.shape[0]
    N = spectrogram.shape[1]
    b = L / M

    sampToRadConst = (
        2.0 * np.pi / L
    )  # Rescale the derivs to rad with step 1 in both directions
    tgradw = a * tgrad * sampToRadConst
    fgradw = (
        -b * (fgrad + np.arange(spectrogram.shape[1]) * a) * sampToRadConst
    )  # also convert relative to freqinv convention

    magnitude_heap = [
        (-max_val, max_pos)
    ]  # Numba requires heap to be initialized with content
    spectrogram[max_pos] = abstol

    small_x, small_y = np.where(spectrogram < max_val * tol)
    for x, y in zip(small_x, small_y):
        spectrogram[x, y] = abstol  # Do not integrate over silence

    while max_val > abstol:
        while (
            len(magnitude_heap) > 0
        ):  # Integrate around maximum value until reaching silence
            max_val, max_pos = heapq.heappop(magnitude_heap)

            col = max_pos[0]
            row = max_pos[1]

            # Spread to 4 direct neighbors
            N_pos = col + 1, row
            S_pos = col - 1, row
            E_pos = col, row + 1
            W_pos = col, row - 1

            if max_pos[0] < M2 - 1 and spectrogram[N_pos] > abstol:
                phase[N_pos] = phase[max_pos] + (fgradw[max_pos] + fgradw[N_pos]) / 2
                heapq.heappush(magnitude_heap, (-spectrogram[N_pos], N_pos))
                spectrogram[N_pos] = abstol

            if max_pos[0] > 0 and spectrogram[S_pos] > abstol:
                phase[S_pos] = phase[max_pos] - (fgradw[max_pos] + fgradw[S_pos]) / 2
                heapq.heappush(magnitude_heap, (-spectrogram[S_pos], S_pos))
                spectrogram[S_pos] = abstol

            if max_pos[1] < N - 1 and spectrogram[E_pos] > abstol:
                phase[E_pos] = phase[max_pos] + (tgradw[max_pos] + tgradw[E_pos]) / 2
                heapq.heappush(magnitude_heap, (-spectrogram[E_pos], E_pos))
                spectrogram[E_pos] = abstol

            if max_pos[1] > 0 and spectrogram[W_pos] > abstol:
                phase[W_pos] = phase[max_pos] - (tgradw[max_pos] + tgradw[W_pos]) / 2
                heapq.heappush(magnitude_heap, (-spectrogram[W_pos], W_pos))
                spectrogram[W_pos] = abstol

        max_val = np.amax(spectrogram)  # Find new maximum value to start integration
        max_x, max_y = np.where(spectrogram == max_val)
        max_pos = max_x[0], max_y[0]
        heapq.heappush(magnitude_heap, (-max_val, max_pos))
        spectrogram[max_pos] = abstol
    return phase

