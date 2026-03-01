"""FFT computation for time-domain waveform analysis.

Uses ``numpy.fft`` (``rfft``/``rfftfreq``) so that existing ``ndarray``
waveform data can be passed directly without any C++ integration.

Window functions are stored in ``WINDOW_REGISTRY`` — a plain ``dict``
mapping each ``WindowFunction`` enum member to a callable that accepts a
length *n* and returns a 1-D numpy weight array.  Adding a new window
requires only a single entry in that dict.

Public API
----------
compute_fft(x, y, window, zero_pad, normalize, output)
    Core computation; returns *(frequencies_hz, values)*.

is_uniform(x, rtol)
    Returns *True* when *x* is sampled on a uniform grid.

resample_uniform(x, y, num_points)
    Re-samples *(x, y)* onto a uniform grid via linear interpolation.

fft_frequency_range(x)
    Returns *(bin_width_hz, nyquist_hz)* for a preview before running the FFT.
"""

import collections.abc
import enum

import numpy as np


class WindowFunction(enum.Enum):
    RECTANGULAR = "Rectangular"
    HAMMING = "Hamming"
    HANNING = "Hanning"
    BLACKMAN = "Blackman"


class FftOutput(enum.Enum):
    MAGNITUDE = "Magnitude"
    MAGNITUDE_DB = "Magnitude (dB)"
    PHASE = "Phase"


class ZeroPadding(enum.Enum):
    NONE = "None"
    NEXT_POWER_OF_TWO = "Next Power of Two"


# ---------------------------------------------------------------------------
# Window function registry — pluggable; add new windows here with one entry.
# Each callable accepts a length n and returns a 1-D numpy weight array.
# ---------------------------------------------------------------------------

WINDOW_REGISTRY: dict[WindowFunction, collections.abc.Callable[[int], np.ndarray]] = {
    WindowFunction.RECTANGULAR: np.ones,
    WindowFunction.HAMMING: np.hamming,
    WindowFunction.HANNING: np.hanning,
    WindowFunction.BLACKMAN: np.blackman,
}


def is_uniform(x: np.ndarray, rtol: float = 1e-3) -> bool:
    """Return *True* when *x* is sampled on a uniform grid within *rtol*.

    Parameters
    ----------
    x    : 1-D array of sample positions (time, etc.).
    rtol : relative tolerance used for the step-size comparison.
    """
    if len(x) < 2:
        return True
    diffs = np.diff(x)
    return bool(np.all(np.isclose(diffs, diffs[0], rtol=rtol)))


def resample_uniform(x: np.ndarray, y: np.ndarray, num_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Re-sample *(x, y)* onto a uniform time grid using linear interpolation.

    Parameters
    ----------
    x          : original (possibly non-uniform) time array.
    y          : corresponding signal array.
    num_points : desired number of output points; defaults to ``len(x)``.

    Returns
    -------
    Tuple *(x_uniform, y_uniform)* on an evenly-spaced grid spanning
    ``[x[0], x[-1]]``.
    """
    if num_points is None:
        num_points = len(x)
    x_uniform = np.linspace(float(x[0]), float(x[-1]), num_points)
    y_uniform = np.interp(x_uniform, x, y)
    return x_uniform, y_uniform


def compute_fft(
    x: np.ndarray,
    y: np.ndarray,
    window: WindowFunction = WindowFunction.RECTANGULAR,
    zero_pad: ZeroPadding = ZeroPadding.NONE,
    normalize: bool = False,
    output: FftOutput = FftOutput.MAGNITUDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FFT of the real-valued signal *y* sampled at times *x*.

    Non-uniform sampling is detected automatically; when present, *y* is
    first re-sampled to a uniform grid via :func:`resample_uniform` before
    windowing and transformation.

    Parameters
    ----------
    x         : time values in seconds (1-D, real, at least 2 elements).
    y         : signal samples (1-D, real-valued, same length as *x*).
    window    : window function applied to the data before the FFT.
    zero_pad  : zero-padding strategy (``NONE`` or ``NEXT_POWER_OF_TWO``).
    normalize : when *True*, scale so that the peak magnitude equals 1.
    output    : which quantity to return (magnitude, dB or phase).

    Returns
    -------
    Tuple *(frequencies, values)* where *frequencies* is in **Hz**.

    Raises
    ------
    ValueError
        If *x* and *y* have different lengths, *y* has fewer than 2 samples,
        or an unrecognised enum value is passed.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(y) < 2:
        raise ValueError("at least 2 samples are required for FFT")
    # ensure real-valued input
    y = np.real(y) if np.iscomplexobj(y) else np.asarray(y, dtype=np.float64)
    # resample to uniform grid when necessary
    if not is_uniform(x):
        x, y = resample_uniform(x, y)
    # sampling interval and rate
    dt = float(x[1] - x[0])
    if dt <= 0:
        raise ValueError("time step dt must be positive")
    n = len(y)
    # apply window function
    win = WINDOW_REGISTRY[window](n)
    y_windowed = y * win
    # zero-padding
    if zero_pad == ZeroPadding.NEXT_POWER_OF_TWO:
        n_fft = int(2 ** np.ceil(np.log2(n)))
    else:
        n_fft = n
    # compute one-sided FFT (rfft discards redundant negative frequencies)
    spectrum = np.fft.rfft(y_windowed, n=n_fft)
    # frequency axis in Hz
    frequencies = np.fft.rfftfreq(n_fft, d=dt)
    # amplitude correction: rfft returns one-sided, so multiply by 2/n
    # (except the DC component at index 0)
    scale = 2.0 / n
    if output == FftOutput.MAGNITUDE:
        values = np.abs(spectrum) * scale
        values[0] /= 2.0
        if normalize:
            peak = float(np.max(values))
            if peak > 0:
                values = values / peak
    elif output == FftOutput.MAGNITUDE_DB:
        magnitude = np.abs(spectrum) * scale
        magnitude[0] /= 2.0
        if normalize:
            peak = float(np.max(magnitude))
            if peak > 0:
                magnitude = magnitude / peak
        # clamp to avoid log(0)
        magnitude = np.maximum(magnitude, 1e-300)
        values = 20.0 * np.log10(magnitude)
    elif output == FftOutput.PHASE:
        values = np.angle(spectrum, deg=True)
    else:
        raise ValueError(f"Unknown FftOutput: {output}")
    return frequencies, values


def fft_frequency_range(x: np.ndarray) -> tuple[float, float]:
    """Return *(bin_width_hz, nyquist_hz)* for a preview.

    Uses the average sampling interval so non-uniform inputs are handled
    gracefully (the result is only approximate for highly non-uniform data).

    Parameters
    ----------
    x : time array (1-D, at least 2 elements).

    Returns
    -------
    Tuple *(bin_width_hz, nyquist_hz)*.  Both values are 0 when the input
    has fewer than 2 elements or the total duration is zero.
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    total_duration = float(x[-1] - x[0])
    if total_duration <= 0:
        return 0.0, 0.0
    dt_avg = total_duration / (n - 1)
    fs = 1.0 / dt_avg
    df = fs / n
    f_nyquist = fs / 2.0
    return df, f_nyquist
