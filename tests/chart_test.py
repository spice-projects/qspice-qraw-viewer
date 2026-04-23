import sys
from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch

import numpy as np

# mock PySide6 submodules before importing chart, which requires Qt at import time
sys.modules.setdefault("PySide6", MagicMock())
sys.modules.setdefault("PySide6.QtCore", MagicMock())
sys.modules.setdefault("PySide6.QtGraphs", MagicMock())
sys.modules.setdefault("PySide6.QtQuick", MagicMock())

from viewer.chart import Chart  # noqa: E402
from viewer.expression import Expression  # noqa: E402


class TestChart(TestCase):

    def test_init_zoom_window(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        # act
        chart = Chart(component, "AC", MagicMock(), abscissa, 10, 90, 1, 500)
        # assert
        self.assertEqual(chart._zoom_window, (10, 0.0, 90, 1.0))

    def test_expressions_initially_empty(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        result = chart.expressions
        # assert
        self.assertEqual(result, [])

    def test_expressions_returns_copy(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        result = chart.expressions
        result.append(MagicMock())
        # assert — internal list must be unaffected by mutation of the returned copy
        self.assertEqual(len(chart.expressions), 0)

    def test_auto_range_returns_early_when_no_series(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        chart.auto_range()
        # assert — no axis interaction when there are no series
        component.createYAxis.assert_not_called()

    def test_selected_steps_setter_noop_when_selection_unchanged(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        chart.plot_series = MagicMock()
        # act
        chart.selected_steps = {0}
        # assert
        chart.plot_series.assert_not_called()

    def test_selected_steps_getter_returns_current_selection(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        # act
        selected_steps = chart.selected_steps
        # assert
        self.assertEqual(selected_steps, {0})

    def test_selected_steps_setter_replots_when_selection_changes(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        chart.plot_series = MagicMock()
        # act
        chart.selected_steps = set()
        # assert
        chart.plot_series.assert_called_once_with([])

    def test_auto_range_calls_set_range_on_axis(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        mock_y_axis = MagicMock()
        vout = Expression("Vout", np.array([1.0, 2.0]), "V")
        # manually inject a series entry with known min/max so auto_range is predictable
        chart._series = {"Vout": (vout, {vout: (mock_y_axis, {}, -1.0, 5.0, "#f77f00")})}
        # act
        chart.auto_range()
        # assert — autorange applies 3% padding on both sides
        mock_y_axis.setRange.assert_called_once_with(-1.18, 5.18)

    def test_auto_range_respects_vertical_zoom(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        mock_y_axis = MagicMock()
        vout = Expression("Vout", np.array([1.0, 2.0]), "V")
        # inject a series with min=-1.0 and max=5.0 (range of 6.0)
        chart._series = {"Vout": (vout, {vout: (mock_y_axis, {}, -1.0, 5.0, "#f77f00")})}
        # apply a vertical zoom that selects only the top quarter (0.0 to 0.25)
        chart.update_zoom_window(-1, -1, 0.0, 0.25)
        # act
        chart.auto_range()
        # assert — autorange uses series min/max and applies 3% padding
        mock_y_axis.setRange.assert_called_with(-1.18, 5.18)

    def test_update_zoom_window_vertical_only(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act — negative x indices signal "no horizontal change"
        chart.update_zoom_window(-1, -1, 0.25, 0.75)
        # assert — only vertical slice of zoom window changed
        self.assertEqual(chart._zoom_window[0], 0)
        self.assertAlmostEqual(chart._zoom_window[1], 0.25)
        self.assertEqual(chart._zoom_window[2], 100)
        self.assertAlmostEqual(chart._zoom_window[3], 0.75)

    def test_update_zoom_window_horizontal_only(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act — None signals "no vertical change"
        chart.update_zoom_window(10, 80, None, None)
        # assert — only horizontal slice of zoom window changed
        self.assertEqual(chart._zoom_window[0], 10)
        self.assertAlmostEqual(chart._zoom_window[1], 0.0)
        self.assertEqual(chart._zoom_window[2], 80)
        self.assertAlmostEqual(chart._zoom_window[3], 1.0)

    def test_update_zoom_window_vertical_zoom_composition(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act — apply two consecutive vertical zooms, each selecting the lower half
        chart.update_zoom_window(-1, -1, 0.0, 0.5)
        chart.update_zoom_window(-1, -1, 0.0, 0.5)
        # assert — second zoom compounds on the first: upper bound goes from 0.5 to 0.25
        self.assertAlmostEqual(chart._zoom_window[1], 0.0)
        self.assertAlmostEqual(chart._zoom_window[3], 0.25)

    def test_reset_zoom_window_vertical(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        chart.reset_zoom_window(-1, -1, 0.1, 0.9)
        # assert — zoom window reflects the provided ratios directly (no composition)
        self.assertAlmostEqual(chart._zoom_window[1], 0.1)
        self.assertAlmostEqual(chart._zoom_window[3], 0.9)

    def test_reset_zoom_window_no_change_when_values_match(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act — reset to the default values that are already in place
        chart.reset_zoom_window(-1, -1, 0.0, 1.0)
        # assert — zoom window unchanged since new values match existing ones
        self.assertAlmostEqual(chart._zoom_window[1], 0.0)
        self.assertAlmostEqual(chart._zoom_window[3], 1.0)

    def test_reset_zoom_window_horizontal(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        chart.reset_zoom_window(20, 70, None, None)
        # assert — horizontal indices updated, vertical unchanged
        self.assertEqual(chart._zoom_window[0], 20)
        self.assertAlmostEqual(chart._zoom_window[1], 0.0)
        self.assertEqual(chart._zoom_window[2], 70)
        self.assertAlmostEqual(chart._zoom_window[3], 1.0)

    def test_get_y_axis_creates_axis_for_new_expression_type(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        axis = chart._get_y_axis("V")
        # assert
        self.assertIsNotNone(axis)
        component.createYAxis.assert_called_once_with(ANY, "V")

    def test_get_y_axis_reuses_existing_axis(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        axis_first = chart._get_y_axis("V")
        # act — request the same expression type a second time
        axis_second = chart._get_y_axis("V")
        # assert — same axis object returned, no extra createYAxis calls
        self.assertIs(axis_first, axis_second)
        self.assertEqual(component.createYAxis.call_count, 1)

    def test_get_y_axis_returns_none_when_four_axes_already_created(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # fill up all four allowed Y axes with distinct expression types
        chart._get_y_axis("V")
        chart._get_y_axis("A")
        chart._get_y_axis("W")
        chart._get_y_axis("s")
        # act — requesting a fifth distinct type must be rejected
        axis = chart._get_y_axis("Hz")
        # assert
        self.assertIsNone(axis)

    def test_clear_resets_internal_state(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        vout = Expression("Vout", np.array([1.0, 2.0]), "V")
        # inject state that clear() must wipe
        chart._y_axes["V"] = MagicMock()
        chart._series["Vout"] = (vout, {})
        # act
        chart.clear()
        # assert — all tracking collections are empty after clear
        self.assertEqual(chart.expressions, [])
        self.assertEqual(chart._series, {})
        self.assertEqual(chart._y_axes, {})

    def test_clear_resets_vertical_zoom_but_preserves_horizontal(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # zoom in horizontally and vertically before clearing
        chart.update_zoom_window(20, 80, None, None)
        chart.update_zoom_window(-1, -1, 0.2, 0.8)
        # act
        chart.clear()
        # assert — vertical zoom is reset to defaults; horizontal range is preserved
        self.assertEqual(chart._zoom_window[0], 20)
        self.assertAlmostEqual(chart._zoom_window[1], 0.0)
        self.assertEqual(chart._zoom_window[2], 80)
        self.assertAlmostEqual(chart._zoom_window[3], 1.0)

    def test_ordinate_values_at_abscissa_value_returns_empty_when_no_series(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        result = chart.ordinate_values_at_abscissa_value(0.5)
        # assert
        self.assertEqual(result, [])

    def test_ordinate_values_at_abscissa_value_returns_name_unit_value_for_plotted_series(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.linspace(0.0, 1.0, 11), "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 11, 1, 500)
        # ordinate: 11 linearly-spaced values from 0 to 100
        vout = Expression("Vout", np.linspace(0.0, 100.0, 11), "V")
        mock_y_axis = MagicMock()
        chart._series["Vout"] = (vout, {vout: (mock_y_axis, {0: MagicMock()}, 0.0, 100.0, "#f77f00")})
        # act — sample at the right edge (x_ratio=1.0) should return the last value
        result = chart.ordinate_values_at_abscissa_value(1.0)
        # assert
        self.assertEqual(len(result), 1)
        name, unit, values = result[0]
        self.assertEqual(name, "Vout")
        self.assertEqual(unit, "V")
        self.assertEqual(values, [100.0])

    def test_ordinate_values_at_abscissa_value_nearest_ordinate_values_at_abscissa_value_midpoint(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.linspace(0.0, 10.0, 11), "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 11, 1, 500)
        # ordinate: index-valued array so we can easily verify which index was sampled
        vout = Expression("Vout", np.arange(11, dtype=float), "V")
        mock_y_axis = MagicMock()
        chart._series["Vout"] = (vout, {vout: (mock_y_axis, {0: MagicMock()}, 0.0, 10.0, "#f77f00")})
        # act — x_ratio maps to the middle x-value of the visible window
        result = chart.ordinate_values_at_abscissa_value(0.5)
        # assert — nearest sample to the midpoint
        _, _, values = result[0]
        self.assertEqual(values, [5.0])

    def test_ordinate_values_at_abscissa_value_clamps_to_zoom_window(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.linspace(0.0, 10.0, 11), "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 2, 8, 1, 500)
        vout = Expression("Vout", np.arange(11, dtype=float), "V")
        mock_y_axis = MagicMock()
        chart._series["Vout"] = (vout, {vout: (mock_y_axis, {0: MagicMock()}, 0.0, 10.0, "#f77f00")})
        # act — x_ratio=0.0 must map to from_index=2, not index 0
        result_left = chart.ordinate_values_at_abscissa_value(0.0)
        # x_ratio=1.0 must map to to_index-1=7
        result_right = chart.ordinate_values_at_abscissa_value(1.0)
        # assert
        _, _, left_val = result_left[0]
        _, _, right_val = result_right[0]
        self.assertEqual(left_val, [2.0])
        self.assertEqual(right_val, [7.0])

    def test_ordinate_values_at_abscissa_value_returns_empty_for_empty_visible_window_even_with_series(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.array([0.0, 1.0, 2.0]), "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 1, 1, 1, 500)
        vout = Expression("Vout", np.array([10.0, 20.0, 30.0]), "V")
        mock_axis = MagicMock()
        chart._series["Vout"] = (vout, {vout: (mock_axis, {0: MagicMock()}, 10.0, 30.0, "#f77f00")})
        # act
        result = chart.ordinate_values_at_abscissa_value(0.5)
        # assert
        self.assertEqual(result, [])

    def test_ordinate_values_at_abscissa_value_falls_back_to_raw_when_cached_points_are_empty(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.array([0.0, 1.0, 2.0, 3.0]), "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 4, 1, 500)
        vout = Expression("Vout", np.array([10.0, 20.0, 30.0, 40.0]), "V")
        mock_axis = MagicMock()
        chart._series["Vout"] = (vout, {vout: (mock_axis, {0: MagicMock()}, 10.0, 40.0, "#f77f00")})
        chart._sample_cache[vout] = {0: (np.array([]), np.array([]))}
        # act
        result = chart.ordinate_values_at_abscissa_value(0.5)
        # assert
        self.assertEqual(result, [("Vout", "V", [20.0])])

    def test_ordinate_values_at_abscissa_value_multiple_series(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.linspace(0.0, 1.0, 5), "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 5, 1, 500)
        vout = Expression("Vout", np.array([10.0, 20.0, 30.0, 40.0, 50.0]), "V")
        iout = Expression("Iout", np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "A")
        mock_axis = MagicMock()
        chart._series["Vout"] = (vout, {vout: (mock_axis, {0: MagicMock()}, 10.0, 50.0, "#f77f00")})
        chart._series["Iout"] = (iout, {iout: (mock_axis, {0: MagicMock()}, 1.0, 5.0, "#00b4d8")})
        # act — x_ratio=0.0 → index 0
        result = chart.ordinate_values_at_abscissa_value(0.0)
        # assert — two entries returned, one per series
        self.assertEqual(len(result), 2)
        names = {r[0] for r in result}
        self.assertIn("Vout", names)
        self.assertIn("Iout", names)

    def test_plot_series_removes_unselected_steps_and_clears_cached_points(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.array([0.0, 1.0, 2.0, 3.0]), "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 4, 2, 500)
        chart._selected_steps = set()
        vout = Expression("Vout", np.array([1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0]), "V")
        rendered_series = {1: MagicMock()}
        mock_axis = MagicMock()
        chart._series["Vout"] = (vout, {vout: (mock_axis, rendered_series, 1.0, 14.0, "#f77f00")})
        chart._sample_cache[vout] = {1: (np.array([0.0]), np.array([1.0]))}
        # act
        chart.plot_series({vout})
        # assert
        self.assertEqual(rendered_series, {})
        self.assertEqual(chart._sample_cache[vout], {})

    def test_plot_series_skips_step_when_decimated_values_are_non_finite(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.array([0.0, 1.0, 2.0, 3.0]), "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 4, 1, 500)
        vout = Expression("Vout", np.array([1.0, 2.0, 3.0, 4.0]), "V")
        # act
        with patch("viewer.chart.decimate_xy", return_value=(np.array([0.0, 1.0]), np.array([np.nan, np.inf]))):
            chart.plot_series({vout})
        # assert
        self.assertIn("Vout", chart._series)
        _, ordinate_series = chart._series["Vout"]
        _, rendered_series, _, _, _ = ordinate_series[vout]
        self.assertEqual(rendered_series, {})
        self.assertNotIn(vout, chart._sample_cache)

    def test_release_y_axis_clears_right_primary_reference(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        axis = MagicMock()
        axis.property.return_value = "V"
        chart._right_y_axis_1 = axis
        chart._y_axes = {"V": axis}
        chart._y_axes_ref_counts = {axis: 1}
        # act
        removed = chart._release_y_axis(axis)
        # assert
        self.assertTrue(removed)
        self.assertIsNone(chart._right_y_axis_1)

    def test_release_y_axis_clears_left_secondary_reference(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        axis = MagicMock()
        axis.property.return_value = "A"
        chart._left_y_axis_2 = axis
        chart._y_axes = {"A": axis}
        chart._y_axes_ref_counts = {axis: 1}
        # act
        removed = chart._release_y_axis(axis)
        # assert
        self.assertTrue(removed)
        self.assertIsNone(chart._left_y_axis_2)

    def test_release_y_axis_clears_right_secondary_reference(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        axis = MagicMock()
        axis.property.return_value = "W"
        chart._right_y_axis_2 = axis
        chart._y_axes = {"W": axis}
        chart._y_axes_ref_counts = {axis: 1}
        # act
        removed = chart._release_y_axis(axis)
        # assert
        self.assertTrue(removed)
        self.assertIsNone(chart._right_y_axis_2)

    def test_ordinate_values_at_abscissa_value_prefers_rendered_decimated_points_over_raw_samples(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.linspace(0.0, 10.0, 11), "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 11, 1, 500)
        # raw data has a narrow spike at x=5 that may be absent from rendered decimated points
        vout = Expression("Vout", np.array([0.0, 0.5, 1.0, 2.0, 3.2, 4.2, 3.1, 2.0, 1.0, 0.5, 0.0]), "V")
        mock_axis = MagicMock()
        chart._series["Vout"] = (vout, {vout: (mock_axis, {0: MagicMock()}, 0.0, 4.2, "#f77f00")})
        # rendered decimated points shown on chart near the same x do not include the raw spike
        chart._sample_cache[vout] = {0: (np.array([0.0, 5.0, 10.0]), np.array([0.0, 3.9, 0.0]))}
        # act
        result = chart.ordinate_values_at_abscissa_value(0.5)
        # assert
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "Vout")
        self.assertEqual(result[0][1], "V")
        self.assertEqual(result[0][2], [3.9])

    def test_abscissa_property(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        result = chart.abscissa
        # assert
        self.assertIs(result, abscissa)

    def test_render_initializes_component_with_abscissa_range(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 10.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        chart.render("Time", "linear", set())
        # assert — initialize must receive label, unit, scale, and exact boundary values
        component.initialize.assert_called_once_with("Time", "s", "linear", float(values[0]), float(values[99]))

    def test_render_calls_auto_range_after_plot_series(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 10.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        with patch.object(chart, "auto_range") as mock_auto_range:
            chart.render("Time", "linear", set())
        # assert
        mock_auto_range.assert_called_once()

    def test_plot_series_adds_expression_to_series_dict(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        decimated_x = np.linspace(0.0, 1.0, 10)
        decimated_y = np.linspace(0.0, 5.0, 10)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(decimated_x, decimated_y)):
            chart.plot_series({vout})
        # assert
        self.assertIn("Vout", chart._series)

    def test_plot_series_stores_expression_in_expressions_list(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        decimated_x = np.linspace(0.0, 1.0, 10)
        decimated_y = np.linspace(0.0, 5.0, 10)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(decimated_x, decimated_y)):
            chart.plot_series({vout})
        # assert
        self.assertIn(vout, chart.expressions)

    def test_plot_series_calls_component_plot_series(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        decimated_x = np.linspace(0.0, 1.0, 10)
        decimated_y = np.linspace(0.0, 5.0, 10)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(decimated_x, decimated_y)):
            chart.plot_series({vout})
        # assert
        component.updateGraphsView.assert_called_once()

    def test_plot_series_skips_already_tracked_expression(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        # inject existing series entry so chart believes Vout is already plotted
        chart._series["Vout"] = (vout, {vout: (MagicMock(), {0: MagicMock()}, 0.0, 5.0, "#f77f00")})
        # act
        chart.plot_series({vout})
        # assert — no new axis was requested because no new series were created
        component.createYAxis.assert_not_called()

    def test_plot_series_removes_expression_absent_from_new_set(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        y_axis = MagicMock()
        y_axis.property.return_value = "V"
        chart._y_axes["V"] = y_axis
        chart._y_axes_ref_counts[y_axis] = 1
        chart._left_y_axis_1 = y_axis
        # inject an existing series that is absent from the new expression set
        chart._series["Vout"] = (vout, {vout: (y_axis, {0: MagicMock()}, 0.0, 5.0, "#f77f00")})
        # act — empty set means all existing series should be removed
        chart.plot_series(set())
        # assert
        self.assertNotIn("Vout", chart._series)
        self.assertNotIn(vout, chart.expressions)
        component.updateGraphsView.assert_called_once()

    def test_plot_series_creates_one_series_entry_per_step(self):
        # arrange
        component = MagicMock()
        n = 10
        values = np.linspace(0.0, 1.0, n)
        abscissa = Expression("Time", values, "s")
        # two steps: 20 ordinate points total (10 per step)
        ordinate_data = np.linspace(0.0, 5.0, 2 * n)
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, n, 2, 500)
        vout = Expression("Vout", ordinate_data, "V")
        decimated_y = np.linspace(0.0, 5.0, n)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(values, decimated_y)):
            chart.plot_series({vout})
        # assert — one QLineSeries created per step
        _, ordinate_series = chart._series["Vout"]
        _, rendered_series, _, _, _ = ordinate_series[vout]
        self.assertEqual(len(rendered_series), 2)

    def test_get_expressions_to_plot_real_returns_single_entry(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        # act
        result = chart._get_expressions_to_plot(vout)
        # assert — real expression returned as-is in a single-element list
        self.assertEqual(result, [vout])

    def test_get_expressions_to_plot_complex_returns_magnitude_and_phase(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        mock_manager = MagicMock()
        magnitude_expr = Expression("db(Vout)", np.ones(10), "dB")
        phase_expr = Expression("phase(Vout)", np.zeros(10), "deg")
        mock_manager.evaluate.side_effect = lambda expr: magnitude_expr if expr == "db(Vout)" else phase_expr
        chart = Chart(component, "AC", mock_manager, abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.ones(10, dtype=np.complex128), "V")
        # act
        result = chart._get_expressions_to_plot(vout)
        # assert — complex expression splits into magnitude then phase
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], magnitude_expr)
        self.assertIs(result[1], phase_expr)
        self.assertEqual(mock_manager.evaluate.call_args_list[0].args, ("db(Vout)",))
        self.assertEqual(mock_manager.evaluate.call_args_list[1].args, ("phase(Vout)",))

    def test_get_expressions_to_plot_complex_returns_empty_when_magnitude_fails(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        mock_manager = MagicMock()
        # evaluate always returns None — magnitude lookup fails immediately
        mock_manager.evaluate.return_value = None
        chart = Chart(component, "AC", mock_manager, abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.ones(10, dtype=np.complex128), "V")
        # act
        result = chart._get_expressions_to_plot(vout)
        # assert
        self.assertEqual(result, [])

    def test_get_expressions_to_plot_complex_returns_empty_when_phase_fails(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        mock_manager = MagicMock()
        magnitude_expr = Expression("db(Vout)", np.ones(10), "dB")
        # magnitude succeeds but phase lookup fails
        mock_manager.evaluate.side_effect = lambda expr: magnitude_expr if expr == "db(Vout)" else None
        chart = Chart(component, "AC", mock_manager, abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.ones(10, dtype=np.complex128), "V")
        # act
        result = chart._get_expressions_to_plot(vout)
        # assert
        self.assertEqual(result, [])

    def test_get_expressions_to_plot_complex_on_tran_chart_returns_empty(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("Time", np.linspace(0.0, 1e-3, 10), "s")
        chart = Chart(component, "TRAN", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.ones(10, dtype=np.complex128), "V")
        # act
        result = chart._get_expressions_to_plot(vout)
        # assert — complex expressions are not plottable in TRAN charts; safe empty list returned
        self.assertEqual(result, [])

    def test_get_expressions_to_plot_complex_on_dc_chart_returns_empty(self):
        # arrange
        component = MagicMock()
        abscissa = Expression("V1", np.linspace(0.0, 5.0, 10), "V")
        chart = Chart(component, "DC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.ones(10, dtype=np.complex128), "V")
        # act
        result = chart._get_expressions_to_plot(vout)
        # assert — complex expressions are not plottable in DC charts; safe empty list returned
        self.assertEqual(result, [])

    def test_clear_resets_axis_slot_references(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        chart._left_y_axis_1 = MagicMock()
        chart._right_y_axis_1 = MagicMock()
        chart._left_y_axis_2 = MagicMock()
        chart._right_y_axis_2 = MagicMock()
        # act
        chart.clear()
        # assert
        self.assertIsNone(chart._left_y_axis_1)
        self.assertIsNone(chart._right_y_axis_1)
        self.assertIsNone(chart._left_y_axis_2)
        self.assertIsNone(chart._right_y_axis_2)

    def test_release_y_axis_returns_false_while_axis_still_shared(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        axis = MagicMock()
        axis.property.return_value = "V"
        chart._left_y_axis_1 = axis
        chart._y_axes["V"] = axis
        chart._y_axes_ref_counts[axis] = 2
        # act
        result = chart._release_y_axis(axis)
        # assert
        self.assertFalse(result)
        self.assertEqual(chart._y_axes_ref_counts[axis], 1)
        self.assertIs(chart._left_y_axis_1, axis)

    def test_release_y_axis_clears_internal_axis_reference_when_unused(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        axis = MagicMock()
        axis.property.return_value = "V"
        chart._left_y_axis_1 = axis
        chart._y_axes["V"] = axis
        chart._y_axes_ref_counts[axis] = 1
        # act
        result = chart._release_y_axis(axis)
        # assert
        self.assertTrue(result)
        self.assertEqual(chart._y_axes, {})
        self.assertEqual(chart._y_axes_ref_counts, {})
        self.assertIsNone(chart._left_y_axis_1)

    def test_plot_series_updates_graphs_view_with_rendered_series(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        decimated_x = np.linspace(0.0, 1.0, 10)
        decimated_y = np.linspace(0.0, 5.0, 10)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(decimated_x, decimated_y)):
            chart.plot_series({vout})
        # assert
        rendered, removed = component.updateGraphsView.call_args.args
        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0][0], "Vout")
        self.assertEqual(removed, [])

    def test_plot_series_sets_series_width_to_two(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        decimated_x = np.linspace(0.0, 1.0, 10)
        decimated_y = np.linspace(0.0, 5.0, 10)
        mock_series = MagicMock()
        # act
        with patch("viewer.chart.QLineSeries", return_value=mock_series):
            with patch("viewer.chart.decimate_xy", return_value=(decimated_x, decimated_y)):
                chart.plot_series({vout})
        # assert
        mock_series.setWidth.assert_called_once_with(2)

    def test_redraw_all_series_calls_resize_abscissa(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 10.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        mock_series = MagicMock()
        chart._series["Vout"] = (vout, {vout: (MagicMock(), {0: mock_series}, 0.0, 5.0, "#f77f00")})
        decimated_y = np.linspace(0.0, 5.0, 10)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(values, decimated_y)):
            chart._redraw_all_series()
        # assert — abscissa axis resized to match the current zoom window bounds
        component.resizeAbscissa.assert_called_once_with(float(values[0]), float(values[-1]))

    def test_redraw_all_series_calls_replace_np_on_each_series(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 10.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.linspace(0.0, 5.0, 10), "V")
        mock_series = MagicMock()
        chart._series["Vout"] = (vout, {vout: (MagicMock(), {0: mock_series}, 0.0, 5.0, "#f77f00")})
        x_out = np.linspace(0.0, 10.0, 10)
        y_out = np.linspace(0.0, 5.0, 10)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(x_out, y_out)):
            chart._redraw_all_series()
        # assert — series data updated with newly decimated arrays
        mock_series.replaceNp.assert_called_once()

    def test_update_zoom_window_horizontal_calls_redraw_all_series(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act
        with patch.object(chart, "_redraw_all_series") as mock_redraw:
            chart.update_zoom_window(10, 80, None, None)
        # assert
        mock_redraw.assert_called_once()

    def test_update_zoom_window_both_calls_redraw_and_auto_range(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        # act — both horizontal and vertical changes supplied simultaneously
        with patch.object(chart, "_redraw_all_series") as mock_redraw:
            with patch.object(chart, "auto_range") as mock_auto_range:
                chart.update_zoom_window(10, 80, 0.25, 0.75)
        # assert
        mock_redraw.assert_called_once()
        mock_auto_range.assert_called_once()

    def test_update_zoom_window_vertical_only_updates_axis_ranges(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        mock_y_axis = MagicMock()
        # inject known axis range so the zoom calculation is predictable
        chart._axis_ranges = {mock_y_axis: (0.0, 10.0)}
        # act — select bottom half (ratios 0.5 → 1.0)
        chart.update_zoom_window(-1, -1, 0.5, 1.0)
        # assert — setRange called with 0+0.5*10=5.0 and 0+1.0*10=10.0
        mock_y_axis.setRange.assert_called_once_with(5.0, 10.0)

    def test_reset_zoom_window_vertical_changed_updates_axis_ranges(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        mock_y_axis = MagicMock()
        chart._axis_ranges = {mock_y_axis: (0.0, 10.0)}
        # act — reset to ratios different from the default (0.0, 1.0) triggers axis update
        chart.reset_zoom_window(-1, -1, 0.3, 0.7)
        # assert — setRange called with raw stored min/max values
        mock_y_axis.setRange.assert_called_once_with(0.0, 10.0)

    def test_reset_zoom_window_no_axis_update_when_vertical_unchanged(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 100, 1, 500)
        mock_y_axis = MagicMock()
        chart._axis_ranges = {mock_y_axis: (0.0, 10.0)}
        # act — same ratios already stored — no change detected
        chart.reset_zoom_window(-1, -1, 0.0, 1.0)
        # assert — setRange must not be called when vertical zoom did not change
        mock_y_axis.setRange.assert_not_called()

    def test_plot_series_handles_constant_nonzero_signal(self):
        # arrange — all ordinate values equal and non-zero: y_range is zero but scale != 0
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.full(10, 3.0), "V")
        decimated_y = np.full(10, 3.0)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(values, decimated_y)):
            chart.plot_series({vout})
        # assert — series stored without raising and min/max reflect the rendered values
        self.assertIn("Vout", chart._series)
        _, ordinate_series = chart._series["Vout"]
        _, _, stored_min, stored_max, _ = ordinate_series[vout]
        self.assertEqual(stored_min, 3.0)
        self.assertEqual(stored_max, 3.0)

    def test_plot_series_handles_all_zero_signal(self):
        # arrange — all ordinate values are zero: scale == 0 triggers the else branch (y_range = 1.0)
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.zeros(10), "V")
        decimated_y = np.zeros(10)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(values, decimated_y)):
            chart.plot_series({vout})
        # assert — series stored and min/max reflect the rendered values
        self.assertIn("Vout", chart._series)
        _, ordinate_series = chart._series["Vout"]
        _, _, stored_min, stored_max, _ = ordinate_series[vout]
        self.assertEqual(stored_min, 0.0)
        self.assertEqual(stored_max, 0.0)

    def test_redraw_all_series_handles_constant_signal(self):
        # arrange — constant ordinate so the flat-signal y_range fix fires inside _redraw_all_series
        component = MagicMock()
        values = np.linspace(0.0, 10.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        vout = Expression("Vout", np.full(10, 7.0), "V")
        mock_series = MagicMock()
        chart._series["Vout"] = (vout, {vout: (MagicMock(), {0: mock_series}, 7.0, 7.0, "#f77f00")})
        decimated_y = np.full(10, 7.0)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(values, decimated_y)):
            chart._redraw_all_series()
        # assert — replaceNp called and resizeAbscissa completed without exception
        mock_series.replaceNp.assert_called_once()
        component.resizeAbscissa.assert_called_once()

    def test_plot_series_skips_expression_when_y_axis_limit_reached(self):
        # arrange — fill up all four Y axis slots before calling plot_series
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 10)
        abscissa = Expression("Time", values, "s")
        chart = Chart(component, "AC", MagicMock(), abscissa, 0, 10, 1, 500)
        # occupy all four axis slots so the fifth unit returns None from _get_y_axis
        chart._get_y_axis("V")
        chart._get_y_axis("A")
        chart._get_y_axis("W")
        chart._get_y_axis("s")
        # fifth expression has a new unit: _get_y_axis will return None
        fifth = Expression("E5", np.linspace(0.0, 1.0, 10), "Hz")
        decimated_y = np.linspace(0.0, 1.0, 10)
        # act
        with patch("viewer.chart.decimate_xy", return_value=(values, decimated_y)):
            chart.plot_series({fifth})
        # assert — expression tracked in _series but its ordinate_series list is empty
        _, ordinate_series = chart._series["E5"]
        self.assertEqual(ordinate_series, {})
