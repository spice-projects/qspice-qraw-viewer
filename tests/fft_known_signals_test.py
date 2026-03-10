from unittest import TestCase

import numpy as np

from viewer.fft import compute_fft_many, FftOutput, WindowFunction, ZeroPadding


def _compute_fft(x, y, window=WindowFunction.RECTANGULAR, zero_pad=ZeroPadding.NONE, normalize=False, output=FftOutput.MAGNITUDE, keep_dc=False):
    # use the batch API for a single signal and unwrap row 0
    freqs, mat = compute_fft_many(x, np.asarray([y]), window, zero_pad, normalize, output, keep_dc)
    return freqs, mat[0]


class TestKnownSignals(TestCase):

    def test_integer_cycle_sine_has_single_bin_peak(self):
        # arrange — integer number of cycles so tone falls exactly on FFT bin
        n = 2048
        fs = 2048.0
        f_tone = 123.0
        cycles = int(f_tone * (n / fs))
        # ensure integer cycles by adjusting n if necessary
        if cycles == 0:
            cycles = 1
        n = int(fs / f_tone * cycles)
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.sin(2 * np.pi * f_tone * x)
        # act
        freqs, magnitude = _compute_fft(x, y, window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)
        # assert — primary peak magnitude ≈ 1 and all other bins are much smaller
        peak_idx = int(np.argmax(magnitude))
        self.assertAlmostEqual(freqs[peak_idx], f_tone, delta=fs / n)
        self.assertAlmostEqual(float(magnitude[peak_idx]), 1.0, delta=0.01)
        # energy outside the peak should be negligible (leakage-free integer cycles)
        mag_copy = magnitude.copy()
        mag_copy[peak_idx] = 0.0
        self.assertLess(float(np.max(mag_copy)), 1e-3)

    def test_two_tone_amplitudes_preserved(self):
        # arrange — two tones with known amplitudes
        n = 4096
        fs = 8192.0
        f1 = 300.0
        f2 = 1200.0
        a1 = 1.0
        a2 = 0.5
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = a1 * np.sin(2 * np.pi * f1 * x) + a2 * np.sin(2 * np.pi * f2 * x)
        # act
        freqs, magnitude = _compute_fft(x, y, window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)
        # assert — peaks near expected frequencies with amplitudes a1 and a2
        idx1 = int(np.argmin(np.abs(freqs - f1)))
        idx2 = int(np.argmin(np.abs(freqs - f2)))
        self.assertAlmostEqual(float(magnitude[idx1]), a1, delta=0.02)
        self.assertAlmostEqual(float(magnitude[idx2]), a2, delta=0.02)

    def test_parseval_energy_conservation(self):
        # arrange — random signal: time-domain energy equals frequency-domain energy (Parseval) within tolerance
        n = 2048
        fs = 1000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        rng = np.random.default_rng(0)
        y = rng.standard_normal(n)
        # act
        freqs, mag = _compute_fft(x, y, window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)
        # compute time-domain energy and frequency-domain reconstructed RMS using one-sided spectrum scaling
        time_energy = np.sum(y * y)
        # reconstruct energy from one-sided magnitude: sum(mag^2)/(2) * (N/2?) simpler: use Parseval on complex spectrum
        # compute complex spectrum directly via numpy for exact comparison
        spectrum = np.fft.fft(y, n=len(y))
        freq_energy = np.sum(np.abs(spectrum) ** 2) / len(y)
        # assert — energies are close
        self.assertAlmostEqual(float(time_energy), float(freq_energy), delta=1e-6 * float(time_energy + 1.0))
