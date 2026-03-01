from unittest import TestCase

import numpy as np

from viewer.fft import (
    FftOutput,
    WindowFunction,
    ZeroPadding,
    WINDOW_REGISTRY,
    compute_fft,
    fft_frequency_range,
    is_uniform,
    resample_uniform,
)


class TestIsUniform(TestCase):

    def test_uniform_grid_returns_true(self):
        x = np.linspace(0.0, 1.0, 1000)
        self.assertTrue(is_uniform(x))

    def test_single_element_returns_true(self):
        self.assertTrue(is_uniform(np.array([0.0])))

    def test_empty_array_returns_true(self):
        self.assertTrue(is_uniform(np.array([])))

    def test_non_uniform_grid_returns_false(self):
        # exponentially spaced — highly non-uniform
        x = np.logspace(0, 3, 100)
        self.assertFalse(is_uniform(x))

    def test_slightly_perturbed_grid_within_tolerance(self):
        x = np.linspace(0.0, 1.0, 100)
        x[50] += 1e-6  # tiny perturbation well within default rtol=1e-3
        self.assertTrue(is_uniform(x, rtol=1e-2))

    def test_slightly_perturbed_grid_outside_tolerance(self):
        x = np.linspace(0.0, 1.0, 100)
        x[50] += 0.05  # large perturbation — exceeds rtol=1e-3
        self.assertFalse(is_uniform(x, rtol=1e-3))


class TestResampleUniform(TestCase):

    def test_output_length_matches_requested(self):
        x = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
        y = np.sin(x)
        x_u, y_u = resample_uniform(x, y, num_points=50)
        self.assertEqual(len(x_u), 50)
        self.assertEqual(len(y_u), 50)

    def test_output_is_uniform(self):
        x = np.logspace(-3, 0, 200)
        y = np.cos(x)
        x_u, _ = resample_uniform(x, y)
        self.assertTrue(is_uniform(x_u))

    def test_endpoints_preserved(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([1.0, 2.0, 3.0])
        x_u, y_u = resample_uniform(x, y, num_points=5)
        self.assertAlmostEqual(float(x_u[0]), 0.0)
        self.assertAlmostEqual(float(x_u[-1]), 1.0)
        self.assertAlmostEqual(float(y_u[0]), 1.0, places=10)
        self.assertAlmostEqual(float(y_u[-1]), 3.0, places=10)

    def test_default_num_points_equals_input_length(self):
        x = np.linspace(0.0, 1.0, 37)
        y = np.ones(37)
        _, y_u = resample_uniform(x, y)
        self.assertEqual(len(y_u), 37)


class TestWindowRegistry(TestCase):

    def test_all_windows_registered(self):
        for wf in WindowFunction:
            self.assertIn(wf, WINDOW_REGISTRY)

    def test_rectangular_is_all_ones(self):
        fn = WINDOW_REGISTRY[WindowFunction.RECTANGULAR]
        win = fn(64)
        np.testing.assert_array_equal(win, np.ones(64))

    def test_hamming_length_correct(self):
        fn = WINDOW_REGISTRY[WindowFunction.HAMMING]
        self.assertEqual(len(fn(128)), 128)

    def test_hanning_length_correct(self):
        fn = WINDOW_REGISTRY[WindowFunction.HANNING]
        self.assertEqual(len(fn(64)), 64)

    def test_blackman_length_correct(self):
        fn = WINDOW_REGISTRY[WindowFunction.BLACKMAN]
        self.assertEqual(len(fn(256)), 256)


class TestComputeFft(TestCase):

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_raises_when_x_and_y_lengths_differ(self):
        x = np.linspace(0.0, 1.0, 10)
        y = np.ones(8)
        with self.assertRaises(ValueError):
            compute_fft(x, y)

    def test_raises_when_fewer_than_two_samples(self):
        x = np.array([0.0])
        y = np.array([1.0])
        with self.assertRaises(ValueError):
            compute_fft(x, y)

    # ------------------------------------------------------------------
    # DC signal
    # ------------------------------------------------------------------

    def test_dc_magnitude_peak_at_zero_hz(self):
        # a pure DC signal must produce a single spike at frequency index 0
        n = 512
        fs = 1000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.ones(n)
        frequencies, magnitude = compute_fft(x, y, output=FftOutput.MAGNITUDE)
        peak_index = int(np.argmax(magnitude))
        self.assertEqual(peak_index, 0)

    # ------------------------------------------------------------------
    # Single-tone signal — magnitude
    # ------------------------------------------------------------------

    def test_single_tone_magnitude_peak_at_correct_bin(self):
        n = 1024
        fs = 10000.0
        f_tone = 1000.0  # 1 kHz sine
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.sin(2 * np.pi * f_tone * x)
        frequencies, magnitude = compute_fft(x, y, window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)
        peak_index = int(np.argmax(magnitude))
        # allow ±1 bin tolerance
        expected_bin = int(round(f_tone / (fs / n)))
        self.assertAlmostEqual(frequencies[peak_index], f_tone, delta=fs / n)

    # ------------------------------------------------------------------
    # Single-tone signal — dB output
    # ------------------------------------------------------------------

    def test_magnitude_db_output_returns_decibels(self):
        n = 512
        fs = 8000.0
        f_tone = 400.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.sin(2 * np.pi * f_tone * x)
        _, magnitude = compute_fft(x, y, output=FftOutput.MAGNITUDE)
        _, db = compute_fft(x, y, output=FftOutput.MAGNITUDE_DB)
        peak = int(np.argmax(magnitude))
        # dB value at the peak should be 20*log10(magnitude) at that bin
        self.assertAlmostEqual(float(db[peak]), 20.0 * np.log10(float(magnitude[peak])), places=4)

    # ------------------------------------------------------------------
    # Phase output
    # ------------------------------------------------------------------

    def test_phase_output_returns_degrees(self):
        n = 512
        fs = 8000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.cos(2 * np.pi * 400.0 * x)
        _, phase = compute_fft(x, y, output=FftOutput.PHASE)
        # phase values must be in [-180, 180]
        self.assertTrue(np.all(phase >= -180.0))
        self.assertTrue(np.all(phase <= 180.0))

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def test_normalize_peaks_at_one(self):
        n = 512
        fs = 4000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = 5.0 * np.sin(2 * np.pi * 500.0 * x)
        _, magnitude = compute_fft(x, y, normalize=True, output=FftOutput.MAGNITUDE)
        self.assertAlmostEqual(float(np.max(magnitude)), 1.0, places=5)

    # ------------------------------------------------------------------
    # Zero padding
    # ------------------------------------------------------------------

    def test_zero_padding_increases_frequency_resolution(self):
        n = 100  # deliberately not a power of two
        fs = 1000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.sin(2 * np.pi * 100.0 * x)
        freq_no_pad, _ = compute_fft(x, y, zero_pad=ZeroPadding.NONE)
        freq_padded, _ = compute_fft(x, y, zero_pad=ZeroPadding.NEXT_POWER_OF_TWO)
        # padded FFT must have more frequency bins
        self.assertGreater(len(freq_padded), len(freq_no_pad))

    # ------------------------------------------------------------------
    # Non-uniform sampling — auto-resampling
    # ------------------------------------------------------------------

    def test_non_uniform_input_is_resampled_without_error(self):
        # exponentially spaced time axis — definitely non-uniform
        x = np.logspace(-4, -1, 200)
        y = np.sin(2 * np.pi * 50.0 * x)
        # must not raise
        frequencies, magnitude = compute_fft(x, y, output=FftOutput.MAGNITUDE)
        self.assertEqual(len(frequencies), len(magnitude))
        self.assertTrue(np.all(np.isfinite(magnitude)))

    # ------------------------------------------------------------------
    # Output shape
    # ------------------------------------------------------------------

    def test_output_arrays_are_same_length(self):
        n = 256
        x = np.linspace(0.0, 1.0, n)
        y = np.random.default_rng(0).standard_normal(n)
        for output in FftOutput:
            with self.subTest(output=output):
                frequencies, values = compute_fft(x, y, output=output)
                self.assertEqual(len(frequencies), len(values))

    # ------------------------------------------------------------------
    # All window functions work end-to-end
    # ------------------------------------------------------------------

    def test_all_window_functions_run_without_error(self):
        n = 256
        x = np.linspace(0.0, 1.0, n)
        y = np.sin(2 * np.pi * 10.0 * x)
        for wf in WindowFunction:
            with self.subTest(window=wf):
                frequencies, values = compute_fft(x, y, window=wf, output=FftOutput.MAGNITUDE)
                self.assertTrue(np.all(np.isfinite(values)))


class TestFftFrequencyRange(TestCase):

    def test_returns_zero_for_fewer_than_two_points(self):
        self.assertEqual(fft_frequency_range(np.array([0.0])), (0.0, 0.0))
        self.assertEqual(fft_frequency_range(np.array([])), (0.0, 0.0))

    def test_nyquist_is_half_sampling_rate(self):
        fs = 1000.0
        n = 1000
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        df, f_nyquist = fft_frequency_range(x)
        self.assertAlmostEqual(f_nyquist, fs / 2, delta=0.1)

    def test_bin_width_equals_fs_over_n(self):
        fs = 2000.0
        n = 500
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        df, _ = fft_frequency_range(x)
        self.assertAlmostEqual(df, fs / n, delta=0.01)

    def test_non_uniform_input_does_not_raise(self):
        x = np.logspace(-3, 0, 100)
        df, f_nyquist = fft_frequency_range(x)
        self.assertGreater(f_nyquist, 0.0)
