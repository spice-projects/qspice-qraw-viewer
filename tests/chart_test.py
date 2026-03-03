import sys
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

# mock PySide6 submodules before importing chart, which requires Qt at import time
sys.modules.setdefault("PySide6", MagicMock())
sys.modules.setdefault("PySide6.QtCore", MagicMock())
sys.modules.setdefault("PySide6.QtGraphs", MagicMock())
sys.modules.setdefault("PySide6.QtQuick", MagicMock())

from viewer.chart import Chart  # noqa: E402
from viewer.variable import Variable, VariableType  # noqa: E402


class TestChart(TestCase):

    def test_init_zoom_window(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        # act
        chart = Chart(component, abscissa, 10, 90, 500)
        # assert
        self.assertEqual(chart._zoom_window, (10, 0.0, 90, 1.0))

    def test_variables_initially_empty(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        # act
        result = chart.variables
        # assert
        self.assertEqual(result, [])

    def test_variables_returns_copy(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        # act
        result = chart.variables
        result.append(MagicMock())
        # assert — internal list must be unaffected by mutation of the returned copy
        self.assertEqual(len(chart.variables), 0)

    def test_auto_range_returns_early_when_no_series(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        # act
        chart.auto_range()
        # assert — no axis interaction when there are no series
        component.createYAxis.assert_not_called()

    def test_auto_range_calls_set_range_on_axis(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        mock_y_axis = MagicMock()
        vout = Variable(1, "Vout", VariableType.VOLTAGE, np.array([1.0, 2.0]), steps=1)
        # manually inject a series entry with known min/max so auto_range is predictable
        chart._series = {"Vout": (vout, [(vout, mock_y_axis, [], -1.0, 5.0)])}
        # act
        chart.auto_range()
        # assert — setRange must be called with the full min/max since zoom is at default (0.0, 1.0)
        mock_y_axis.setRange.assert_called_once_with(-1.0, 5.0)

    def test_auto_range_respects_vertical_zoom(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        mock_y_axis = MagicMock()
        vout = Variable(1, "Vout", VariableType.VOLTAGE, np.array([1.0, 2.0]), steps=1)
        # inject a series with min=-1.0 and max=5.0 (range of 6.0)
        chart._series = {"Vout": (vout, [(vout, mock_y_axis, [], -1.0, 5.0)])}
        # apply a vertical zoom that selects only the top quarter (0.0 to 0.25)
        chart.update_zoom_window(-1, -1, 0.0, 0.25)
        # act
        chart.auto_range()
        # assert — range = 6.0, top 25% means upper bound = -1.0 + 0.25 * 6.0 = 0.5
        mock_y_axis.setRange.assert_called_with(-1.0, 0.5)

    def test_update_zoom_window_vertical_only(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
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
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
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
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
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
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        # act
        chart.reset_zoom_window(-1, -1, 0.1, 0.9)
        # assert — zoom window reflects the provided ratios directly (no composition)
        self.assertAlmostEqual(chart._zoom_window[1], 0.1)
        self.assertAlmostEqual(chart._zoom_window[3], 0.9)

    def test_reset_zoom_window_no_change_when_values_match(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        # act — reset to the default values that are already in place
        chart.reset_zoom_window(-1, -1, 0.0, 1.0)
        # assert — zoom window unchanged since new values match existing ones
        self.assertAlmostEqual(chart._zoom_window[1], 0.0)
        self.assertAlmostEqual(chart._zoom_window[3], 1.0)

    def test_reset_zoom_window_horizontal(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        # act
        chart.reset_zoom_window(20, 70, None, None)
        # assert — horizontal indices updated, vertical unchanged
        self.assertEqual(chart._zoom_window[0], 20)
        self.assertAlmostEqual(chart._zoom_window[1], 0.0)
        self.assertEqual(chart._zoom_window[2], 70)
        self.assertAlmostEqual(chart._zoom_window[3], 1.0)

    def test_get_y_axis_creates_axis_for_new_variable_type(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        # act
        axis = chart._get_y_axis(VariableType.VOLTAGE)
        # assert
        self.assertIsNotNone(axis)
        component.createYAxis.assert_called_once_with("Y Axis 1", "V")

    def test_get_y_axis_reuses_existing_axis(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        axis_first = chart._get_y_axis(VariableType.VOLTAGE)
        # act — request the same variable type a second time
        axis_second = chart._get_y_axis(VariableType.VOLTAGE)
        # assert — same axis object returned, no extra createYAxis calls
        self.assertIs(axis_first, axis_second)
        self.assertEqual(component.createYAxis.call_count, 1)

    def test_get_y_axis_returns_none_when_four_axes_already_created(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        # fill up all four allowed Y axes with distinct variable types
        chart._get_y_axis(VariableType.VOLTAGE)
        chart._get_y_axis(VariableType.CURRENT)
        chart._get_y_axis(VariableType.POWER)
        chart._get_y_axis(VariableType.TIME)
        # act — requesting a fifth distinct type must be rejected
        axis = chart._get_y_axis(VariableType.FREQUENCY)
        # assert
        self.assertIsNone(axis)

    def test_clear_resets_internal_state(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
        vout = Variable(1, "Vout", VariableType.VOLTAGE, np.array([1.0, 2.0]), steps=1)
        # inject state that clear() must wipe
        chart._variables.append(vout)
        chart._y_axes[VariableType.VOLTAGE] = MagicMock()
        chart._series["Vout"] = (vout, [])
        # act
        chart.clear()
        # assert — all tracking collections are empty after clear
        self.assertEqual(chart._variables, [])
        self.assertEqual(chart._series, {})
        self.assertEqual(chart._y_axes, {})

    def test_clear_resets_vertical_zoom_but_preserves_horizontal(self):
        # arrange
        component = MagicMock()
        values = np.linspace(0.0, 1.0, 100)
        abscissa = Variable(0, "Time", VariableType.TIME, values, steps=1)
        chart = Chart(component, abscissa, 0, 100, 500)
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
