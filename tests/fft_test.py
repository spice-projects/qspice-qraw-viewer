from unittest import TestCase

import numpy as np

from viewer.fft import WINDOW_REGISTRY, FftOutput, WindowFunction, ZeroPadding, compute_fft, fft_frequency_range, is_uniform, resample_uniform


class TestFft(TestCase):

    def test_is_uniform_uniform_grid_returns_true(self):
        # arrange
        x = np.linspace(0.0, 1.0, 1000)
        # act
        result = is_uniform(x)
        # assert
        self.assertTrue(result)

    def test_is_uniform_single_element_returns_true(self):
        # arrange
        x = np.array([0.0])
        # act
        result = is_uniform(x)
        # assert
        self.assertTrue(result)

    def test_is_uniform_empty_array_returns_true(self):
        # arrange
        x = np.array([])
        # act
        result = is_uniform(x)
        # assert
        self.assertTrue(result)

    def test_is_uniform_non_uniform_grid_returns_false(self):
        # arrange — exponentially spaced, highly non-uniform
        x = np.logspace(0, 3, 100)
        # act
        result = is_uniform(x)
        # assert
        self.assertFalse(result)

    def test_is_uniform_small_perturbation_within_tolerance(self):
        # arrange — tiny perturbation well within default rtol=1e-3
        x = np.linspace(0.0, 1.0, 100)
        x[50] += 1e-6
        # act
        result = is_uniform(x, rtol=1e-2)
        # assert
        self.assertTrue(result)

    def test_is_uniform_large_perturbation_outside_tolerance(self):
        # arrange — large perturbation that exceeds rtol=1e-3
        x = np.linspace(0.0, 1.0, 100)
        x[50] += 0.05
        # act
        result = is_uniform(x, rtol=1e-3)
        # assert
        self.assertFalse(result)

    def test_resample_uniform_output_length_matches_requested(self):
        # arrange
        x = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
        y = np.sin(x)
        # act
        x_u, y_u = resample_uniform(x, y, num_points=50)
        # assert
        self.assertEqual(len(x_u), 50)
        self.assertEqual(len(y_u), 50)

    def test_resample_uniform_output_is_uniform(self):
        # arrange
        x = np.logspace(-3, 0, 200)
        y = np.cos(x)
        # act
        x_u, _ = resample_uniform(x, y)
        # assert
        self.assertTrue(is_uniform(x_u))

    def test_resample_uniform_endpoints_preserved(self):
        # arrange
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([1.0, 2.0, 3.0])
        # act
        x_u, y_u = resample_uniform(x, y, num_points=5)
        # assert
        self.assertAlmostEqual(float(x_u[0]), 0.0)
        self.assertAlmostEqual(float(x_u[-1]), 1.0)
        self.assertAlmostEqual(float(y_u[0]), 1.0, places=10)
        self.assertAlmostEqual(float(y_u[-1]), 3.0, places=10)

    def test_resample_uniform_default_num_points_equals_input_length(self):
        # arrange
        x = np.linspace(0.0, 1.0, 37)
        y = np.ones(37)
        # act
        _, y_u = resample_uniform(x, y)
        # assert
        self.assertEqual(len(y_u), 37)

    def test_window_registry_all_windows_registered(self):
        # arrange
        # (no extra setup — iterating over the enum is sufficient)
        # act / assert
        for wf in WindowFunction:
            self.assertIn(wf, WINDOW_REGISTRY)

    def test_window_registry_rectangular_is_all_ones(self):
        # arrange
        fn = WINDOW_REGISTRY[WindowFunction.RECTANGULAR]
        # act
        win = fn(64)
        # assert
        np.testing.assert_array_equal(win, np.ones(64))

    def test_window_registry_hamming_length_correct(self):
        # arrange
        fn = WINDOW_REGISTRY[WindowFunction.HAMMING]
        # act
        win = fn(128)
        # assert
        self.assertEqual(len(win), 128)

    def test_window_registry_hanning_length_correct(self):
        # arrange
        fn = WINDOW_REGISTRY[WindowFunction.HANNING]
        # act
        win = fn(64)
        # assert
        self.assertEqual(len(win), 64)

    def test_window_registry_blackman_length_correct(self):
        # arrange
        fn = WINDOW_REGISTRY[WindowFunction.BLACKMAN]
        # act
        win = fn(256)
        # assert
        self.assertEqual(len(win), 256)

    def test_compute_fft_raises_when_x_and_y_lengths_differ(self):
        # arrange
        x = np.linspace(0.0, 1.0, 10)
        y = np.ones(8)
        # act / assert
        with self.assertRaises(ValueError):
            compute_fft(x, y)

    def test_compute_fft_raises_when_fewer_than_two_samples(self):
        # arrange
        x = np.array([0.0])
        y = np.array([1.0])
        # act / assert
        with self.assertRaises(ValueError):
            compute_fft(x, y)

    def test_compute_fft_dc_magnitude_peak_at_zero_hz(self):
        # arrange — pure DC signal must produce a spike at frequency index 0
        n = 512
        fs = 1000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.ones(n)
        # act
        frequencies, magnitude = compute_fft(x, y, output=FftOutput.MAGNITUDE)
        # assert
        peak_index = int(np.argmax(magnitude))
        self.assertEqual(peak_index, 0)

    def test_compute_fft_single_tone_magnitude_peak_at_correct_bin(self):
        # arrange
        n = 1024
        fs = 10000.0
        f_tone = 1000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.sin(2 * np.pi * f_tone * x)
        # act
        frequencies, magnitude = compute_fft(x, y, window=WindowFunction.RECTANGULAR, output=FftOutput.MAGNITUDE)
        # assert — allow ±1 bin tolerance
        peak_index = int(np.argmax(magnitude))
        self.assertAlmostEqual(frequencies[peak_index], f_tone, delta=fs / n)

    def test_compute_fft_magnitude_db_output_returns_decibels(self):
        # arrange
        n = 512
        fs = 8000.0
        f_tone = 400.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.sin(2 * np.pi * f_tone * x)
        # act
        _, magnitude = compute_fft(x, y, output=FftOutput.MAGNITUDE)
        _, db = compute_fft(x, y, output=FftOutput.MAGNITUDE_DB)
        # assert — dB at the peak must equal 20*log10(linear magnitude)
        peak = int(np.argmax(magnitude))
        self.assertAlmostEqual(float(db[peak]), 20.0 * np.log10(float(magnitude[peak])), places=4)

    def test_compute_fft_phase_output_returns_degrees(self):
        # arrange
        n = 512
        fs = 8000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.cos(2 * np.pi * 400.0 * x)
        # act
        _, phase = compute_fft(x, y, output=FftOutput.PHASE)
        # assert
        self.assertTrue(np.all(phase >= -180.0))
        self.assertTrue(np.all(phase <= 180.0))

    def test_compute_fft_normalize_peaks_at_one(self):
        # arrange
        n = 512
        fs = 4000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = 5.0 * np.sin(2 * np.pi * 500.0 * x)
        # act
        _, magnitude = compute_fft(x, y, normalize=True, output=FftOutput.MAGNITUDE)
        # assert
        self.assertAlmostEqual(float(np.max(magnitude)), 1.0, places=5)

    def test_compute_fft_zero_padding_increases_bin_count(self):
        # arrange — deliberately non-power-of-two length
        n = 100
        fs = 1000.0
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        y = np.sin(2 * np.pi * 100.0 * x)
        # act
        freq_no_pad, _ = compute_fft(x, y, zero_pad=ZeroPadding.NONE)
        freq_padded, _ = compute_fft(x, y, zero_pad=ZeroPadding.NEXT_POWER_OF_TWO)
        # assert — padded FFT must have more frequency bins
        self.assertGreater(len(freq_padded), len(freq_no_pad))

    def test_compute_fft_non_uniform_input_is_resampled_without_error(self):
        # arrange — exponentially spaced time axis, definitely non-uniform
        x = np.logspace(-4, -1, 200)
        y = np.sin(2 * np.pi * 50.0 * x)
        # act
        frequencies, magnitude = compute_fft(x, y, output=FftOutput.MAGNITUDE)
        # assert
        self.assertEqual(len(frequencies), len(magnitude))
        self.assertTrue(np.all(np.isfinite(magnitude)))

    def test_compute_fft_output_arrays_are_same_length(self):
        # arrange
        n = 256
        x = np.linspace(0.0, 1.0, n)
        y = np.random.default_rng(0).standard_normal(n)
        # act / assert — verified for all output types
        for output in FftOutput:
            with self.subTest(output=output):
                frequencies, values = compute_fft(x, y, output=output)
                self.assertEqual(len(frequencies), len(values))

    def test_compute_fft_all_window_functions_run_without_error(self):
        # arrange
        n = 256
        x = np.linspace(0.0, 1.0, n)
        y = np.sin(2 * np.pi * 10.0 * x)
        # act / assert — verified for all window functions
        for wf in WindowFunction:
            with self.subTest(window=wf):
                frequencies, values = compute_fft(x, y, window=wf, output=FftOutput.MAGNITUDE)
                self.assertTrue(np.all(np.isfinite(values)))

    def test_fft_frequency_range_returns_zero_for_fewer_than_two_points(self):
        # arrange
        x_one = np.array([0.0])
        x_empty = np.array([])
        # act
        result_one = fft_frequency_range(x_one)
        result_empty = fft_frequency_range(x_empty)
        # assert
        self.assertEqual(result_one, (0.0, 0.0))
        self.assertEqual(result_empty, (0.0, 0.0))

    def test_fft_frequency_range_nyquist_is_half_sampling_rate(self):
        # arrange
        fs = 1000.0
        n = 1000
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        # act
        df, f_nyquist = fft_frequency_range(x)
        # assert
        self.assertAlmostEqual(f_nyquist, fs / 2, delta=0.1)

    def test_fft_frequency_range_bin_width_equals_fs_over_n(self):
        # arrange
        fs = 2000.0
        n = 500
        x = np.linspace(0.0, n / fs, n, endpoint=False)
        # act
        df, _ = fft_frequency_range(x)
        # assert
        self.assertAlmostEqual(df, fs / n, delta=0.01)

    def test_fft_frequency_range_non_uniform_input_does_not_raise(self):
        # arrange
        x = np.logspace(-3, 0, 100)
        # act
        df, f_nyquist = fft_frequency_range(x)
        # assert
        self.assertGreater(f_nyquist, 0.0)
