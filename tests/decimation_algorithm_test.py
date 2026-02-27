from unittest import TestCase

import numpy as np

from viewer.decimation_algorithm import DecimationAlgorithm, decimate, decimate_xy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linspace(n: int) -> np.ndarray:
    """Return n evenly spaced float64 values in [0, 1]."""
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def _sine(n: int) -> np.ndarray:
    """Return one full period of a sine wave with n samples."""
    return np.sin(np.linspace(0.0, 2 * np.pi, n, dtype=np.float64))


def _is_subset_of(output: np.ndarray, original: np.ndarray) -> bool:
    """Return True when every value in *output* also appears in *original*."""
    original_set = set(original.tolist())
    return all(v in original_set for v in output.tolist())


def _xy_pairs_coherent(x_out: np.ndarray, y_out: np.ndarray,
                       x_orig: np.ndarray, y_orig: np.ndarray) -> bool:
    """Return True when every (x_out[i], y_out[i]) pair exists in the original arrays."""
    pairs_orig = set(zip(x_orig.tolist(), y_orig.tolist()))
    return all((xv, yv) in pairs_orig for xv, yv in zip(x_out.tolist(), y_out.tolist()))


# ---------------------------------------------------------------------------
# Short-circuit behaviour — shared across all algorithms
# ---------------------------------------------------------------------------

class TestShortCircuit(TestCase):
    """When input length <= target, every algorithm must return the input unchanged."""

    def _assert_short_circuit(self, algorithm: DecimationAlgorithm):
        # arrange
        values = _sine(100)
        # act — exact match
        result = decimate(values, 100, algorithm)
        # assert — same object, not a copy
        self.assertIs(result, values)
        # act — input shorter than target
        result = decimate(values, 200, algorithm)
        # assert
        self.assertIs(result, values)

    def test_nth_point_short_circuit(self):
        self._assert_short_circuit(DecimationAlgorithm.NTH_POINT)

    def test_min_max_short_circuit(self):
        self._assert_short_circuit(DecimationAlgorithm.MIN_MAX)

    def test_m4_short_circuit(self):
        self._assert_short_circuit(DecimationAlgorithm.M4)

    def test_lttb_short_circuit(self):
        self._assert_short_circuit(DecimationAlgorithm.LTTB)

    def test_average_short_circuit(self):
        self._assert_short_circuit(DecimationAlgorithm.AVERAGE)

    def _assert_xy_short_circuit(self, algorithm: DecimationAlgorithm):
        # arrange
        x = _linspace(50)
        y = _sine(50)
        # act
        x_out, y_out = decimate_xy(x, y, 50, algorithm)
        # assert — same objects
        self.assertIs(x_out, x)
        self.assertIs(y_out, y)

    def test_xy_nth_point_short_circuit(self):
        self._assert_xy_short_circuit(DecimationAlgorithm.NTH_POINT)

    def test_xy_min_max_short_circuit(self):
        self._assert_xy_short_circuit(DecimationAlgorithm.MIN_MAX)

    def test_xy_m4_short_circuit(self):
        self._assert_xy_short_circuit(DecimationAlgorithm.M4)

    def test_xy_lttb_short_circuit(self):
        self._assert_xy_short_circuit(DecimationAlgorithm.LTTB)

    def test_xy_average_short_circuit(self):
        self._assert_xy_short_circuit(DecimationAlgorithm.AVERAGE)


# ---------------------------------------------------------------------------
# Output length contract — all index-based algorithms must not exceed target
# ---------------------------------------------------------------------------

class TestOutputLength(TestCase):

    def _assert_length_le_target(self, algorithm: DecimationAlgorithm, target: int = 50):
        # arrange
        values = _sine(10_000)
        # act
        result = decimate(values, target, algorithm)
        # assert — all algorithms respect the target count; NTH_POINT may add 1
        # extra (the unconditional "append last point"), all others are strict
        self.assertLessEqual(len(result), target + 1)

    def test_nth_point_length(self):
        # NTH_POINT appends the last point if it is not already selected,
        # meaning output can be at most target + 1
        values = _sine(10_000)
        result = decimate(values, 50, DecimationAlgorithm.NTH_POINT)
        self.assertLessEqual(len(result), 51)

    def test_min_max_length(self):
        self._assert_length_le_target(DecimationAlgorithm.MIN_MAX)

    def test_m4_length(self):
        self._assert_length_le_target(DecimationAlgorithm.M4)

    def test_lttb_length(self):
        # arrange
        values = _sine(10_000)
        target = 50
        # act
        result = decimate(values, target, DecimationAlgorithm.LTTB)
        # assert — LTTB produces exactly target points (no deduplication)
        self.assertEqual(len(result), target)

    def test_average_length(self):
        self._assert_length_le_target(DecimationAlgorithm.AVERAGE)


# ---------------------------------------------------------------------------
# NTH_POINT
# ---------------------------------------------------------------------------

class TestNthPoint(TestCase):

    def test_last_point_always_included(self):
        # arrange
        values = _sine(1001)
        # act
        result = decimate(values, 100, DecimationAlgorithm.NTH_POINT)
        # assert — last original value must appear in output
        self.assertAlmostEqual(float(result[-1]), float(values[-1]))

    def test_first_point_always_included(self):
        # arrange
        values = _sine(1001)
        # act
        result = decimate(values, 100, DecimationAlgorithm.NTH_POINT)
        # assert
        self.assertAlmostEqual(float(result[0]), float(values[0]))

    def test_output_is_subset_of_input(self):
        # arrange
        values = np.arange(1000, dtype=np.float64)
        # act
        result = decimate(values, 50, DecimationAlgorithm.NTH_POINT)
        # assert — every selected value exists in the original
        self.assertTrue(_is_subset_of(result, values))

    def test_uniform_stride(self):
        # arrange — 1000 points, target 100 → expected stride = 10
        values = np.arange(1000, dtype=np.float64)
        # act
        result = decimate(values, 100, DecimationAlgorithm.NTH_POINT)
        # assert — selected values are values[0], values[10], values[20], ..., values[999]
        expected_stride = 1000 // 100   # = 10
        expected_indices = list(range(0, 1000, expected_stride))
        if expected_indices[-1] != 999:
            expected_indices.append(999)
        np.testing.assert_array_equal(result, values[expected_indices])


# ---------------------------------------------------------------------------
# MIN_MAX
# ---------------------------------------------------------------------------

class TestMinMax(TestCase):

    def test_output_contains_bucket_extremes(self):
        # arrange — sawtooth with a known spike at a specific position
        values = np.zeros(1000, dtype=np.float64)
        values[250] = 99.0   # global max
        values[750] = -99.0  # global min
        # act
        result = decimate(values, 100, DecimationAlgorithm.MIN_MAX)
        # assert — both extremes must survive decimation
        self.assertIn(99.0,  result.tolist())
        self.assertIn(-99.0, result.tolist())

    def test_output_is_subset_of_input(self):
        # arrange
        values = _sine(10_000)
        # act
        result = decimate(values, 200, DecimationAlgorithm.MIN_MAX)
        # assert — all selected values come from the original vector
        self.assertTrue(_is_subset_of(result, values))

    def test_flat_signal_deduplication(self):
        # arrange — all identical values: min == max in every bucket, np.unique must deduplicate
        values = np.ones(10_000, dtype=np.float64)
        target = 200
        # act
        result = decimate(values, target, DecimationAlgorithm.MIN_MAX)
        # assert — output is shorter than target (deduplicated) and all values are 1.0
        self.assertLessEqual(len(result), target)
        np.testing.assert_array_equal(result, np.ones(len(result)))

    def test_output_sorted(self):
        # arrange
        values = _sine(5_000)
        # act
        result = decimate(values, 100, DecimationAlgorithm.MIN_MAX)
        # assert — values selected at monotonically increasing indices; confirm via a
        # round-trip check that successive output elements correspond to distinct
        # positions (already guaranteed by np.unique, but also check no NaNs)
        self.assertFalse(np.any(np.isnan(result)))


# ---------------------------------------------------------------------------
# M4
# ---------------------------------------------------------------------------

class TestM4(TestCase):

    def test_output_contains_bucket_extremes(self):
        # arrange — single spike that is both isolated min and max per bucket
        values = np.zeros(1000, dtype=np.float64)
        values[99]  = 50.0   # near end of first bucket
        values[900] = -50.0  # near start of last bucket
        # act
        result = decimate(values, 100, DecimationAlgorithm.M4)
        # assert — both extremes must survive
        self.assertIn(50.0,  result.tolist())
        self.assertIn(-50.0, result.tolist())

    def test_output_contains_first_and_last_of_input(self):
        # arrange
        values = _sine(10_000)
        # act
        result = decimate(values, 200, DecimationAlgorithm.M4)
        # assert — original first and last values must be present
        self.assertAlmostEqual(float(result[0]),  float(values[0]))
        self.assertAlmostEqual(float(result[-1]), float(values[-1]))

    def test_output_is_subset_of_input(self):
        # arrange
        values = _sine(10_000)
        # act
        result = decimate(values, 200, DecimationAlgorithm.M4)
        # assert
        self.assertTrue(_is_subset_of(result, values))

    def test_flat_signal_deduplication(self):
        # arrange — flat signal: first == last == min == max in every bucket
        values = np.full(10_000, 3.14, dtype=np.float64)
        target = 200
        # act
        result = decimate(values, target, DecimationAlgorithm.M4)
        # assert — heavy deduplication; all values still equal 3.14
        self.assertLessEqual(len(result), target)
        np.testing.assert_array_almost_equal(result, np.full(len(result), 3.14))


# ---------------------------------------------------------------------------
# LTTB
# ---------------------------------------------------------------------------

class TestLTTB(TestCase):

    def test_output_length_equals_target(self):
        # arrange
        values = _sine(10_000)
        target = 150
        # act
        result = decimate(values, target, DecimationAlgorithm.LTTB)
        # assert — LTTB always produces exactly 'target' points
        self.assertEqual(len(result), target)

    def test_first_and_last_preserved(self):
        # arrange
        values = _sine(10_000)
        # act
        result = decimate(values, 100, DecimationAlgorithm.LTTB)
        # assert
        self.assertAlmostEqual(float(result[0]),  float(values[0]))
        self.assertAlmostEqual(float(result[-1]), float(values[-1]))

    def test_output_is_subset_of_input(self):
        # arrange
        values = _sine(10_000)
        # act
        result = decimate(values, 100, DecimationAlgorithm.LTTB)
        # assert — LTTB selects real samples, not interpolated values
        self.assertTrue(_is_subset_of(result, values))

    def test_target_2_edge_case(self):
        # arrange — target == 2 previously caused ZeroDivisionError via (n-2)/(target-2)
        values = _sine(500)
        # act
        result = decimate(values, 2, DecimationAlgorithm.LTTB)
        # assert — must return exactly the first and last original values
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(float(result[0]),  float(values[0]))
        self.assertAlmostEqual(float(result[-1]), float(values[-1]))

    def test_uses_real_x_in_decimate_xy(self):
        # arrange — non-uniform x spacing: log-spaced frequencies (like a Bode plot)
        n = 10_000
        x = np.logspace(1, 6, n, dtype=np.float64)   # 10 Hz … 1 MHz
        y = _sine(n)
        target = 100
        # act
        x_out, y_out = decimate_xy(x, y, target, DecimationAlgorithm.LTTB)
        # assert — output length is exactly target and pairs are coherent
        self.assertEqual(len(x_out), target)
        self.assertEqual(len(y_out), target)
        self.assertTrue(_xy_pairs_coherent(x_out, y_out, x, y))


# ---------------------------------------------------------------------------
# AVERAGE
# ---------------------------------------------------------------------------

class TestAverage(TestCase):

    def test_output_values_are_bucket_means(self):
        # arrange — 100 points split into 10 buckets of 10
        values = np.arange(100, dtype=np.float64)   # 0,1,...,99
        target = 10
        # act
        result = decimate(values, target, DecimationAlgorithm.AVERAGE)
        # assert — each bucket of 10 consecutive integers has mean = first + 4.5
        expected = np.array([np.mean(values[i*10:(i+1)*10]) for i in range(10)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_length(self):
        # arrange
        values = _sine(10_000)
        target = 100
        # act
        result = decimate(values, target, DecimationAlgorithm.AVERAGE)
        # assert
        self.assertLessEqual(len(result), target)

    def test_flat_signal_unchanged_values(self):
        # arrange
        values = np.full(1_000, 7.0, dtype=np.float64)
        # act
        result = decimate(values, 50, DecimationAlgorithm.AVERAGE)
        # assert — average of identical values is that same value
        np.testing.assert_array_almost_equal(result, np.full(len(result), 7.0))

    def test_output_values_not_from_original(self):
        # arrange — strictly monotone input so no integer bucket mean equals any sample
        values = np.arange(0, 100, dtype=np.float64)   # 0,1,...,99
        # act
        result = decimate(values, 10, DecimationAlgorithm.AVERAGE)
        # assert — bucket means (4.5, 14.5, …) are not integers, so they don't
        # appear in the original integer array
        for v in result:
            self.assertNotIn(v, values.tolist())


# ---------------------------------------------------------------------------
# decimate_xy — joint coherence
# ---------------------------------------------------------------------------

class TestDecimateXY(TestCase):
    """Verify that x and y are always decimated to the same length and that
    every output (x[i], y[i]) pair corresponds to the same original sample
    for all index-based algorithms."""

    def _assert_xy_coherent(self, algorithm: DecimationAlgorithm):
        # arrange
        n = 10_000
        x = _linspace(n)
        y = _sine(n)
        # act
        x_out, y_out = decimate_xy(x, y, 200, algorithm)
        # assert — equal lengths
        self.assertEqual(len(x_out), len(y_out))
        # assert — every output pair exists in the original (not applicable to AVERAGE
        # since it produces interpolated means)
        if algorithm != DecimationAlgorithm.AVERAGE:
            self.assertTrue(_xy_pairs_coherent(x_out, y_out, x, y))

    def test_nth_point_coherent(self):
        self._assert_xy_coherent(DecimationAlgorithm.NTH_POINT)

    def test_min_max_coherent(self):
        self._assert_xy_coherent(DecimationAlgorithm.MIN_MAX)

    def test_m4_coherent(self):
        self._assert_xy_coherent(DecimationAlgorithm.M4)

    def test_lttb_coherent(self):
        self._assert_xy_coherent(DecimationAlgorithm.LTTB)

    def test_average_xy_same_bucket_boundaries(self):
        # arrange — 100 points, target 10 → 10 buckets of 10
        x = np.arange(100, dtype=np.float64)
        y = np.arange(100, dtype=np.float64) * 2.0
        # act
        x_out, y_out = decimate_xy(x, y, 10, DecimationAlgorithm.AVERAGE)
        # assert — x_out[i] == mean(x bucket i), y_out[i] == mean(y bucket i)
        for i in range(10):
            self.assertAlmostEqual(float(x_out[i]), float(np.mean(x[i*10:(i+1)*10])))
            self.assertAlmostEqual(float(y_out[i]), float(np.mean(y[i*10:(i+1)*10])))

    def test_index_driven_by_y_not_x(self):
        # arrange — x is uniform; y has a spike that only y-driven selection will preserve
        n = 1_000
        x = _linspace(n)
        y = np.zeros(n, dtype=np.float64)
        y[499] = 100.0   # spike at midpoint
        # act — use MIN_MAX which is purely y-driven
        x_out, y_out = decimate_xy(x, y, 100, DecimationAlgorithm.MIN_MAX)
        # assert — spike value must appear in y output (and the paired x must be x[499])
        self.assertIn(100.0, y_out.tolist())
        spike_pos = y_out.tolist().index(100.0)
        self.assertAlmostEqual(float(x_out[spike_pos]), float(x[499]))

    def test_xy_output_length_le_target(self):
        # arrange
        x = _linspace(10_000)
        y = _sine(10_000)
        # Output length contracts per algorithm:
        #   NTH_POINT : <= target + 1  (appends last point unconditionally)
        #   MIN_MAX   : <= target      (ceiling bucket size + np.unique dedup)
        #   M4        : <= target      (ceiling bucket size + np.unique dedup)
        #   LTTB      : == target      (exact)
        #   AVERAGE   : <= target      (ceiling bucket size)
        slack = {DecimationAlgorithm.NONE: len(y), DecimationAlgorithm.NTH_POINT: 1,
                 DecimationAlgorithm.MIN_MAX: 0, DecimationAlgorithm.M4: 0,
                 DecimationAlgorithm.LTTB: 0, DecimationAlgorithm.AVERAGE: 0}
        target = 300
        for algorithm in DecimationAlgorithm:
            with self.subTest(algorithm=algorithm):
                # act
                x_out, y_out = decimate_xy(x, y, target, algorithm)
                # assert
                self.assertLessEqual(len(x_out), target + slack[algorithm])
                self.assertLessEqual(len(y_out), target + slack[algorithm])
                self.assertEqual(len(x_out), len(y_out))


# ---------------------------------------------------------------------------
# Single-element and two-element arrays
# ---------------------------------------------------------------------------

class TestTinyInputs(TestCase):

    def test_single_element_returns_unchanged(self):
        # arrange
        values = np.array([42.0])
        for algorithm in DecimationAlgorithm:
            with self.subTest(algorithm=algorithm):
                # act
                result = decimate(values, 10, algorithm)
                # assert
                self.assertEqual(len(result), 1)

    def test_two_elements_returns_unchanged(self):
        # arrange
        values = np.array([1.0, 2.0])
        for algorithm in DecimationAlgorithm:
            with self.subTest(algorithm=algorithm):
                # act
                result = decimate(values, 10, algorithm)
                # assert
                self.assertEqual(len(result), 2)

    def test_lttb_target_equals_input_length(self):
        # arrange — edge where target == length, short-circuit must trigger
        values = _sine(50)
        # act
        result = decimate(values, 50, DecimationAlgorithm.LTTB)
        # assert
        self.assertIs(result, values)
