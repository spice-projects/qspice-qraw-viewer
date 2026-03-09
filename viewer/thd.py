"""THD computation helpers.

Provides a single helper ``compute_thd`` which calculates Total Harmonic
Distortion (THD) from FFT outputs (frequency bins and linear magnitudes).

This module is purposely small and independent so it can be tested in
isolation and later wired into the application UI if desired.
"""
from __future__ import annotations

import numpy as np


def compute_thd(frequencies: np.ndarray, values: np.ndarray, fundamental_freq: float | None = None, n_harmonics: int = 5, tol_hz: float | None = None) -> float:
    """Compute Total Harmonic Distortion (THD) from an FFT result.

    Parameters
    ----------
    frequencies     : 1-D array of frequency bin centres in Hz (as returned by
                      :func:`np.fft.rfftfreq`).
    values          : 1-D array of linear magnitudes for each frequency bin
                      (as returned by an FFT magnitude output).
    fundamental_freq: optional known fundamental frequency in Hz. When
                      ``None`` the function picks the largest non-DC peak as
                      the fundamental.
    n_harmonics     : number of harmonics to include (k = 2..n_harmonics).
    tol_hz          : acceptance tolerance in Hz for locating harmonic bins.
                      When ``None`` a default of half a bin width is used.

    Returns
    -------
    THD ratio as a floating-point value (linear, e.g. 0.1 means 10%).

    Raises
    ------
    ValueError
        If the fundamental amplitude is zero or input arrays are malformed.
    """
    # basic validation
    if len(frequencies) != len(values) or len(values) == 0:
        raise ValueError("frequencies and values must be non-empty arrays of the same length")
    # determine bin width and default tolerance
    df = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.0
    if tol_hz is None:
        tol_hz = max(0.5 * df, 1e-12)
    # find fundamental index
    if fundamental_freq is None:
        # exclude DC (index 0) when searching for the fundamental
        if len(values) < 2:
            raise ValueError("not enough frequency bins to determine fundamental")
        funda_idx = int(np.argmax(values[1:]) + 1)
        funda_freq = float(frequencies[funda_idx])
    else:
        funda_idx = int(np.argmin(np.abs(frequencies - float(fundamental_freq))))
        funda_freq = float(frequencies[funda_idx])
    # fundamental amplitude
    A1 = float(values[funda_idx])
    if A1 <= 0.0:
        raise ValueError("fundamental amplitude is zero or non-positive")
    # sum squared amplitudes of harmonics (k=2..n_harmonics)
    sum_sq = 0.0
    nyquist = float(frequencies[-1])
    for k in range(2, int(n_harmonics) + 1):
        target = k * funda_freq
        # stop if harmonic is above Nyquist
        if target > nyquist:
            break
        # find bins within tolerance of target frequency
        idxs = np.where(np.abs(frequencies - target) <= tol_hz)[0]
        if idxs.size == 0:
            # if no bin within tol, pick nearest bin if it's reasonably close
            nearest = int(np.argmin(np.abs(frequencies - target)))
            if abs(frequencies[nearest] - target) <= 2.0 * tol_hz:
                ak = float(values[nearest])
            else:
                # skip this harmonic
                continue
        else:
            ak = float(np.max(values[idxs]))
        sum_sq += ak * ak
    # compute THD ratio (linear)
    thd = float(np.sqrt(sum_sq) / A1)
    return thd
