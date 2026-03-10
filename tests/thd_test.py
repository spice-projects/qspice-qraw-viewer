from unittest import TestCase

import numpy as np

from viewer.fft import FftOutput, WindowFunction, ZeroPadding, compute_fft_many
from viewer.thd import compute_thd


def _compute_fft(x, y, window=WindowFunction.RECTANGULAR, zero_pad=ZeroPadding.NONE, normalize=False, output=FftOutput.MAGNITUDE, keep_dc=False):
    # use the batch API for a single signal and unwrap row 0
    frequencies, values_matrix = compute_fft_many(x, np.asarray([y]), window, zero_pad, normalize, output, keep_dc)
    return frequencies, values_matrix[0]


class TestThd(TestCase):

    def test_compute_thd_simple_second_harmonic(self):
        # arrange — fundamental with a small second harmonic (A1=1.0, A2=0.1)
        n = 2048
        fs = 8192.0
        f1 = 124.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = 1.0 * np.sin(2 * np.pi * f1 * x) + 0.1 * np.sin(2 * np.pi * 2.0 * f1 * x)
        # act — compute FFT and then THD
        freqs, magnitude = _compute_fft(x, y, window=WindowFunction.RECTANGULAR, zero_pad=ZeroPadding.NEXT_POWER_OF_TWO, output=FftOutput.MAGNITUDE)
        thd = compute_thd(freqs, magnitude, n_harmonics=2)
        # assert — expected THD = 0.1 (linear)
        self.assertAlmostEqual(float(thd), 0.1, places=2)

    def test_compute_thd_multiple_harmonics(self):
        # arrange — fundamental plus 2nd and 3rd harmonics (A1=1, A2=0.2, A3=0.05)
        n = 2048
        fs = 8192.0
        f1 = 48.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = 1.0 * np.sin(2 * np.pi * f1 * x) + 0.2 * np.sin(2 * np.pi * 2.0 * f1 * x) + 0.05 * np.sin(2 * np.pi * 3.0 * f1 * x)
        # act
        freqs, magnitude = _compute_fft(x, y, window=WindowFunction.HANNING, zero_pad=ZeroPadding.NEXT_POWER_OF_TWO, output=FftOutput.MAGNITUDE)
        thd = compute_thd(freqs, magnitude, n_harmonics=3)
        # expected THD = sqrt(0.2^2 + 0.05^2) / 1.0
        expected = float((0.2 ** 2 + 0.05 ** 2) ** 0.5)
        # assert — allow a small tolerance due to windowing/leakage
        self.assertAlmostEqual(float(thd), expected, delta=0.01)

    def test_compute_thd_raises_on_zero_fundamental(self):
        # arrange — zero signal so fundamental amplitude is zero
        n = 256
        x = np.linspace(0.0, 1.0, n, endpoint=False)
        y = np.zeros(n)
        freqs, magnitude = _compute_fft(x, y, window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)
        # act / assert
        with self.assertRaises(ValueError):
            compute_thd(freqs, magnitude)

    def test_compute_thd_pure_sine_no_distortion(self):
        # arrange — pure sine with frequency aligned to FFT bins should yield no harmonics
        n = 2048
        fs = 8192.0
        # choose f1 such that it is exactly an integer number of cycles in the window
        f1 = fs * 10.0 / n
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.sin(2 * np.pi * f1 * x)
        # act — compute FFT and then THD
        freqs, magnitude = _compute_fft(x, y, window=WindowFunction.RECTANGULAR, zero_pad=ZeroPadding.NEXT_POWER_OF_TWO, output=FftOutput.MAGNITUDE)
        thd = compute_thd(freqs, magnitude, n_harmonics=5)
        # assert — THD should be effectively zero (allow tiny numerical tolerance)
        self.assertAlmostEqual(float(thd), 0.0, delta=1e-6)
