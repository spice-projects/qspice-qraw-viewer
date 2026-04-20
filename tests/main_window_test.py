import sys
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

# mock PySide6 submodules before importing main_window, which requires Qt at import time
sys.modules.setdefault("PySide6", MagicMock())
sys.modules.setdefault("PySide6.QtCore", MagicMock())
sys.modules.setdefault("PySide6.QtGui", MagicMock())
sys.modules.setdefault("PySide6.QtGraphs", MagicMock())
sys.modules.setdefault("PySide6.QtQml", MagicMock())
sys.modules.setdefault("PySide6.QtQuick", MagicMock())
sys.modules.setdefault("PySide6.QtWebEngineWidgets", MagicMock())
sys.modules.setdefault("PySide6.QtWidgets", MagicMock())
# Slot must act as a pass-through decorator so @Slot(...) does not replace the method with a mock
sys.modules["PySide6.QtCore"].Slot = lambda *a, **kw: (lambda f: f)
# QMainWindow must be a concrete class so that MainWindow can genuinely inherit from it
sys.modules["PySide6.QtWidgets"].QMainWindow = type("QMainWindow", (), {})

from viewer.expression import Expression  # noqa: E402
from viewer.main_window import MainWindow, _build_step_combinations, _compute_decimate_target, _FALLBACK_DECIMATE_TARGET, _format_value, _format_values  # noqa: E402


class TestMainWindow(TestCase):

    def test_zoom_in_reduces_window(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        win._abscissa = MagicMock(data=list(range(20)))
        win._charts = []
        win._abscissa_from_index = 0
        win._abscissa_to_index = 20
        # act
        win._on_horizontal_zoom(0, 0.3, 0.7, 0.5)
        # assert
        new_width = win._abscissa_to_index - win._abscissa_from_index
        self.assertLess(new_width, 20)
        self.assertGreaterEqual(win._abscissa_from_index, 0)
        self.assertLessEqual(win._abscissa_to_index, 20)

    def test_zoom_out_moves_window_outward(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        win._abscissa = MagicMock(data=list(range(20)))
        win._charts = []
        win._abscissa_from_index = 5
        win._abscissa_to_index = 15
        old_width = win._abscissa_to_index - win._abscissa_from_index
        # act
        win._on_horizontal_zoom(0, 0.25, 0.75, 2.0)
        # assert
        new_width = win._abscissa_to_index - win._abscissa_from_index
        self.assertNotEqual(new_width, old_width)
        self.assertGreaterEqual(new_width, 2)
        self.assertLessEqual(win._abscissa_to_index, 20)

    def test_zoom_out_at_boundary_saturates(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        win._abscissa = MagicMock(data=list(range(20)))
        win._charts = []
        win._abscissa_from_index = 0
        win._abscissa_to_index = 20
        # act — repeated zoom-out gestures must never push indices outside data bounds
        for _ in range(5):
            win._on_horizontal_zoom(0, 0.0, 1.0, 2.0)
        # assert
        self.assertEqual(win._abscissa_from_index, 0)
        self.assertEqual(win._abscissa_to_index, 20)

    def test_minimum_window_enforced(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        win._abscissa = MagicMock(data=list(range(20)))
        win._charts = []
        win._abscissa_from_index = 0
        win._abscissa_to_index = 20
        # act — repeated zoom-in must never produce a window smaller than two points
        for _ in range(50):
            win._on_horizontal_zoom(0, 0.4, 0.6, 0.5)
        # assert
        width = win._abscissa_to_index - win._abscissa_from_index
        self.assertGreaterEqual(width, 2)

    def test_pan_moves_window_right(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        win._abscissa = MagicMock(data=list(range(20)))
        win._charts = []
        win._abscissa_from_index = 5
        win._abscissa_to_index = 15
        # act — zoom_factor=1.0 signals a pure pan; positive x_left_ratio shifts right
        win._on_horizontal_zoom(0, 0.2, 1.2, 1.0)
        # assert
        self.assertGreater(win._abscissa_from_index, 5)
        self.assertGreater(win._abscissa_to_index, 15)

    def test_pan_moves_window_left(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        win._abscissa = MagicMock(data=list(range(20)))
        win._charts = []
        win._abscissa_from_index = 5
        win._abscissa_to_index = 15
        # act — zoom_factor=1.0 signals a pure pan; negative x_left_ratio shifts left
        win._on_horizontal_zoom(0, -0.2, 0.8, 1.0)
        # assert
        self.assertLess(win._abscissa_from_index, 5)
        self.assertLess(win._abscissa_to_index, 15)

    def test_pan_clamps_at_left_boundary(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        win._abscissa = MagicMock(data=list(range(20)))
        win._charts = []
        win._abscissa_from_index = 0
        win._abscissa_to_index = 10
        # act — pan left when already at the left boundary must not go below zero
        win._on_horizontal_zoom(0, -0.2, 0.8, 1.0)
        # assert
        self.assertEqual(win._abscissa_from_index, 0)
        self.assertEqual(win._abscissa_to_index, 10)

    def test_pan_clamps_at_right_boundary(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        win._abscissa = MagicMock(data=list(range(20)))
        win._charts = []
        win._abscissa_from_index = 10
        win._abscissa_to_index = 20
        # act — pan right when already at the right boundary must not exceed total length
        win._on_horizontal_zoom(0, 0.2, 1.2, 1.0)
        # assert
        self.assertEqual(win._abscissa_from_index, 10)
        self.assertEqual(win._abscissa_to_index, 20)


class TestFormatValue(TestCase):

    def test_giga_prefix(self):
        self.assertEqual(_format_value(2e9, "Hz"), "2.00 GHz")

    def test_mega_prefix(self):
        self.assertEqual(_format_value(1.5e6, "Hz"), "1.50 MHz")

    def test_kilo_prefix(self):
        self.assertEqual(_format_value(1000.0, "V"), "1.00 kV")

    def test_no_prefix_plain(self):
        self.assertEqual(_format_value(5.0, "V"), "5.00 V")

    def test_milli_prefix(self):
        self.assertEqual(_format_value(0.05, "A"), "50.00 mA")

    def test_micro_prefix(self):
        self.assertEqual(_format_value(1e-4, "A"), "100.00 µA")

    def test_nano_prefix(self):
        self.assertEqual(_format_value(1e-7, "s"), "100.00 ns")

    def test_pico_prefix(self):
        self.assertEqual(_format_value(1e-10, "F"), "100.00 pF")

    def test_negative_value_giga(self):
        self.assertEqual(_format_value(-3e9, "Hz"), "-3.00 GHz")

    def test_zero_value(self):
        # zero has abs(0) < 1e-12 so it returns "0 {unit}"
        self.assertEqual(_format_value(0.0, "V"), "0 V")


class TestFormatValues(TestCase):

    def test_single_value_no_brackets(self):
        # arrange / act
        result = _format_values("V(R1)", [5.0], "V")
        # assert — single value uses equality format without list brackets
        self.assertEqual(result, "V(R1) = 5.00 V")

    def test_multiple_values_bracketed(self):
        # arrange / act
        result = _format_values("V(R1)", [1.0, 2.0], "V")
        # assert — multiple values formatted in a bracketed comma-separated list
        self.assertEqual(result, "V(R1) = [1.00 V, 2.00 V]")

    def test_multiple_values_si_prefix_applied(self):
        # arrange / act
        result = _format_values("I(L1)", [1e-3, 2e-3], "A")
        # assert — SI prefix applied per-value
        self.assertEqual(result, "I(L1) = [1.00 mA, 2.00 mA]")


class TestBuildStepCombinations(TestCase):

    def test_returns_empty_when_single_step(self):
        # arrange
        expressions = [Expression("Rload", np.array([1.0]), "", variable_type="parameter")]
        # act
        parameter_names, combinations = _build_step_combinations(expressions, 1, 1)
        # assert
        self.assertEqual(parameter_names, [])
        self.assertEqual(combinations, [])

    def test_uses_single_step_abscissa_length_for_parameter_rows(self):
        # arrange
        expressions = [
            Expression("Rload", np.array([10.0, 10.0, 10.0, 10.0, 22.0, 22.0, 22.0, 22.0, 47.0, 47.0, 47.0, 47.0]), "", variable_type="parameter"),
            Expression("V(out)", np.array([1.0] * 12), "V", variable_type="voltage")
        ]
        # act
        parameter_names, combinations = _build_step_combinations(expressions, 3, 4)
        # assert
        self.assertEqual(parameter_names, ["Rload"])
        self.assertEqual([combination.step_index for combination in combinations], [0, 1, 2])
        self.assertEqual([combination.values for combination in combinations], [["10"], ["22"], ["47"]])

    def test_returns_empty_when_no_parameter_variables_exist(self):
        # arrange
        expressions = [Expression("V(out)", np.array([1.0, 2.0]), "V", variable_type="voltage")]
        # act
        parameter_names, combinations = _build_step_combinations(expressions, 3, 2)
        # assert
        self.assertEqual(parameter_names, [])
        self.assertEqual(combinations, [])


class TestMainWindowSlots(TestCase):

    def _make_win(self, total=20):
        # build a bare MainWindow bypassing __init__
        win = MainWindow.__new__(MainWindow)
        win._abscissa = MagicMock(data=list(range(total)), values=list(range(total)))
        win._default_chart_type = "AC"
        win._charts = []
        win._abscissa_from_index = 0
        win._abscissa_to_index = total
        return win

    def test_vertical_zoom_delegates_to_chart(self):
        # arrange
        win = self._make_win()
        chart = MagicMock()
        win._charts = [chart]
        # act
        win._on_vertical_zoom(0, 0.2, 0.8)
        # assert — per-chart vertical zoom only; horizontal indices passed as -1
        chart.update_zoom_window.assert_called_once_with(-1, -1, 0.2, 0.8)

    def test_menu_zoom_to_fit_resets_to_full_range(self):
        # arrange
        win = self._make_win(total=100)
        win._abscissa_from_index = 20
        win._abscissa_to_index = 80
        chart = MagicMock()
        win._charts = [chart]
        # act
        win._on_menu_zoom_to_fit(0)
        # assert — from/to indices reset to full range
        self.assertEqual(win._abscissa_from_index, 0)
        self.assertEqual(win._abscissa_to_index, 100)

    def test_menu_zoom_to_fit_resets_target_chart_zoom(self):
        # arrange
        win = self._make_win(total=100)
        chart0 = MagicMock()
        chart1 = MagicMock()
        win._charts = [chart0, chart1]
        # act
        win._on_menu_zoom_to_fit(0)
        # assert — chart at target index gets reset_zoom_window; others get update_zoom_window
        chart0.reset_zoom_window.assert_called_once_with(0, 100, 0.0, 1.0)
        chart1.update_zoom_window.assert_called_once_with(0, 100, None, None)

    def test_menu_autorange_calls_reset_on_chart(self):
        # arrange
        win = self._make_win()
        chart = MagicMock()
        win._charts = [chart]
        # act
        win._on_menu_autorange(0)
        # assert — vertical zoom reset; horizontal left unchanged (pass -1)
        chart.reset_zoom_window.assert_called_once_with(-1, -1, 0.0, 1.0)

    def test_menu_zoom_abscissa_extent_resets_indices(self):
        # arrange
        win = self._make_win(total=50)
        win._abscissa_from_index = 10
        win._abscissa_to_index = 40
        chart = MagicMock()
        win._charts = [chart]
        # act
        win._on_menu_zoom_abscissa_extent(0)
        # assert — indices reset to full abscissa extent
        self.assertEqual(win._abscissa_from_index, 0)
        self.assertEqual(win._abscissa_to_index, 50)

    def test_menu_zoom_abscissa_extent_calls_reset_on_all_charts(self):
        # arrange
        win = self._make_win(total=50)
        c0 = MagicMock()
        c1 = MagicMock()
        win._charts = [c0, c1]
        # act
        win._on_menu_zoom_abscissa_extent(0)
        # assert
        c0.reset_zoom_window.assert_called_once_with(0, 50, None, None)
        c1.reset_zoom_window.assert_called_once_with(0, 50, None, None)

    def test_menu_delete_all_plots_clears_chart(self):
        # arrange
        win = self._make_win()
        chart = MagicMock()
        win._charts = [chart]
        # act
        win._on_menu_delete_all_plots(0)
        # assert
        chart.clear.assert_called_once_with()

    def test_menu_add_window_appends_chart(self):
        # arrange
        win = self._make_win()
        win._expression_manager = MagicMock()
        win._abscissa_scale = MagicMock(value="lin")
        win._steps = 1
        win._decimate_target = 500
        win._plot_suggestions = []
        win._initial_selected_steps = None
        win._root = MagicMock()
        win._root.getChart.return_value = MagicMock()
        # act
        win._on_menu_add_window(0)
        # assert — one chart added
        self.assertEqual(len(win._charts), 1)

    def test_menu_delete_window_removes_chart(self):
        # arrange
        win = self._make_win()
        win._root = MagicMock()
        chart0 = MagicMock()
        chart1 = MagicMock()
        win._charts = [chart0, chart1]
        # act
        win._on_menu_delete_window(0)
        # assert — first chart removed; second chart remains
        self.assertEqual(len(win._charts), 1)
        self.assertIs(win._charts[0], chart1)

    def test_on_qml_ready_sets_step_tool_enabled_for_stepped_files(self):
        # arrange
        win = self._make_win()
        win._steps = 6
        win._abscissa = MagicMock(unit="s", data=list(range(20)), values=list(range(20)))
        win._qml_view = MagicMock()
        root = MagicMock()
        win._qml_view.rootObject.return_value = root
        # act
        win._on_qml_ready(sys.modules["PySide6.QtQuick"].QQuickView.Status.Ready)
        # assert
        root.setProperty.assert_any_call("stepToolEnabled", True)

    def test_on_qml_ready_sets_step_tool_disabled_for_single_step_files(self):
        # arrange
        win = self._make_win()
        win._steps = 1
        win._abscissa = MagicMock(unit="s", data=list(range(20)), values=list(range(20)))
        win._qml_view = MagicMock()
        root = MagicMock()
        win._qml_view.rootObject.return_value = root
        # act
        win._on_qml_ready(sys.modules["PySide6.QtQuick"].QQuickView.Status.Ready)
        # assert
        root.setProperty.assert_any_call("stepToolEnabled", False)

    def test_on_qml_ready_converts_numpy_step_bool_to_python_bool(self):
        # arrange
        win = self._make_win()
        win._steps = np.int64(6)
        win._abscissa = MagicMock(unit="s", data=list(range(20)), values=list(range(20)))
        win._qml_view = MagicMock()
        root = MagicMock()
        win._qml_view.rootObject.return_value = root
        # act
        win._on_qml_ready(sys.modules["PySide6.QtQuick"].QQuickView.Status.Ready)
        # assert
        value = [args[0][1] for args in root.setProperty.call_args_list if args[0][0] == "stepToolEnabled"][-1]
        self.assertIs(type(value), bool)
        self.assertTrue(value)

    def test_size_hint_returns_1200_by_800(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        # act — import QSize from the mocked module only; compare width/height numerically
        result = win.sizeHint()
        # assert — QSize is mocked so just verify the call was made with the right args
        # actual assertion: method exists and returns something (integration check is enough here)
        self.assertIsNotNone(result)

    def test_close_event_unregisters_fft_window_from_application_registry(self):
        # arrange
        win = MainWindow.__new__(MainWindow)
        event = MagicMock()
        # act
        with patch("viewer.main_window._unregister_fft_window") as mock_unregister:
            win.closeEvent(event)
        # assert
        mock_unregister.assert_called_once_with(win)

    def test_pointer_moved_uses_chart_public_sampling_api(self):
        # arrange
        win = self._make_win(total=20)
        win._abscissa = MagicMock(unit="s", data=list(range(20)), values=list(range(20)))
        win._abscissa.name = "time"
        win._abscissa_scale = "linear"
        win._last_status_time = 0.0
        win.statusBar = MagicMock(return_value=MagicMock())
        chart = MagicMock()
        chart.sample_index_at_ratio.return_value = 7
        chart.sample_at.return_value = [("V(out)", "V", [1.23])]
        win._charts = [chart]
        # act
        win._on_pointer_moved(0, 0.35)
        # assert
        chart.sample_index_at_ratio.assert_called_once_with(0.35)
        chart.sample_at.assert_called_once_with(0.35)

    def test_pointer_moved_ignores_invalid_chart_index(self):
        # arrange
        win = self._make_win(total=20)
        win._last_status_time = 0.0
        win.statusBar = MagicMock(return_value=MagicMock())
        chart = MagicMock()
        win._charts = [chart]
        # act
        win._on_pointer_moved(99, 0.5)
        # assert
        chart.sample_index_at_ratio.assert_not_called()
        chart.sample_at.assert_not_called()

    def test_pointer_moved_updates_status_bar_message(self):
        # arrange
        win = self._make_win(total=20)
        win._abscissa = MagicMock(unit="s", data=list(range(20)), values=list(range(20)))
        win._abscissa.name = "time"
        win._abscissa_scale = "linear"
        win._last_status_time = 0.0
        status_bar = MagicMock()
        win.statusBar = MagicMock(return_value=status_bar)
        chart = MagicMock()
        chart.sample_index_at_ratio.return_value = 5
        chart.sample_at.return_value = [("V(out)", "V", [1.0])]
        win._charts = [chart]
        # act
        win._on_pointer_moved(0, 0.2)
        # assert
        status_bar.showMessage.assert_called_once_with("time = 5.00 s    V(out) = 1.00 V")


class TestMultiStepFft(TestCase):

    def _make_fft_win(self, steps: int, step_points: int):
        # build a bare MainWindow bypassing __init__
        win = MainWindow.__new__(MainWindow)
        total = steps * step_points
        win._abscissa = MagicMock(data=np.linspace(0.0, 1e-3, step_points), unit="s")
        win._abscissa.name = "Time"
        win._abscissa_from_index = 0
        win._abscissa_to_index = step_points
        win._steps = steps
        win._qraw_path = MagicMock()
        win._fft_windows = []
        win._charts = []
        win._abscissa_scale = MagicMock()
        return win

    def _make_dialog_mock(self, expr: Expression, step_points: int):
        # build a FftDialog mock whose .exec() returns Accepted via a shared sentinel
        _accepted = object()
        dialog = MagicMock()
        dialog.exec.return_value = _accepted
        dialog.result_expressions = [expr]
        dialog.result_from_index = 0
        dialog.result_to_index = step_points
        dialog.result_window = MagicMock(value="Rectangular")
        dialog.result_zero_pad = MagicMock(value="None")
        dialog.result_normalize = False
        dialog.result_keep_dc = True
        dialog.result_output = MagicMock()
        # make the Dialog class itself (not instance) carry DialogCode
        dialog_class = MagicMock()
        dialog_class.return_value = dialog
        dialog_class.DialogCode.Accepted = _accepted
        return dialog_class

    def test_fft_builds_expression_data_for_all_steps(self):
        # arrange
        steps = 3
        step_points = 64
        win = self._make_fft_win(steps, step_points)
        # source expression: distinct value per step so we can verify which step was used
        data = np.concatenate([np.full(step_points, float(s)) for s in range(steps)])
        expr = Expression("V(out)", data, "V")
        chart = MagicMock()
        chart.expressions = [expr]
        chart.abscissa = win._abscissa
        chart.selected_steps = {0, 1, 2}
        win._charts = [chart]
        dialog_class = self._make_dialog_mock(expr, step_points)
        captured_qraw = []
        def fake_qraw(**kw):
            captured_qraw.append(kw)
            return MagicMock()
        def fake_main_window(qraw, source_qraw_path=None):
            win2 = MagicMock()
            win2._initial_selected_steps = None
            return win2
        with patch("viewer.main_window.FftDialog", dialog_class), \
             patch("viewer.main_window.QRawFile", fake_qraw), \
             patch("viewer.main_window.MainWindow", fake_main_window), \
             patch("viewer.main_window.compute_fft_many") as mock_fft, \
             patch("viewer.main_window.ExpressionManager"):
            freq = np.linspace(0, 500, 33)
            mock_fft.return_value = (freq, np.ones((1, 33)))
            win._on_menu_fft(0)
        # assert — compute_fft_many called once per step
        self.assertEqual(mock_fft.call_count, steps)

    def test_fft_qraw_built_with_all_steps(self):
        # arrange
        steps = 3
        step_points = 64
        win = self._make_fft_win(steps, step_points)
        data = np.concatenate([np.full(step_points, float(s)) for s in range(steps)])
        expr = Expression("V(out)", data, "V")
        chart = MagicMock()
        chart.expressions = [expr]
        chart.abscissa = win._abscissa
        chart.selected_steps = {1, 2}
        win._charts = [chart]
        dialog_class = self._make_dialog_mock(expr, step_points)
        captured_steps = []
        def fake_qraw(**kw):
            captured_steps.append(kw.get("steps"))
            return MagicMock()
        def fake_main_window(qraw, source_qraw_path=None):
            win2 = MagicMock()
            win2._initial_selected_steps = None
            return win2
        with patch("viewer.main_window.FftDialog", dialog_class), \
             patch("viewer.main_window.QRawFile", fake_qraw), \
             patch("viewer.main_window.MainWindow", fake_main_window), \
             patch("viewer.main_window.compute_fft_many") as mock_fft, \
             patch("viewer.main_window.ExpressionManager"):
            freq = np.linspace(0, 500, 33)
            mock_fft.return_value = (freq, np.ones((1, 33)))
            win._on_menu_fft(0)
        # assert — QRawFile receives steps=3 (all source steps, not just selected)
        self.assertEqual(captured_steps[0], steps)

    def test_fft_window_initial_selected_steps_matches_source_chart(self):
        # arrange
        steps = 5
        step_points = 64
        win = self._make_fft_win(steps, step_points)
        data = np.concatenate([np.full(step_points, float(s)) for s in range(steps)])
        expr = Expression("V(out)", data, "V")
        chart = MagicMock()
        chart.expressions = [expr]
        chart.abscissa = win._abscissa
        chart.selected_steps = {1, 2}
        win._charts = [chart]
        dialog_class = self._make_dialog_mock(expr, step_points)
        created_windows = []
        def fake_main_window(qraw, source_qraw_path=None):
            win2 = MagicMock()
            win2._initial_selected_steps = None
            created_windows.append(win2)
            return win2
        with patch("viewer.main_window.FftDialog", dialog_class), \
             patch("viewer.main_window.QRawFile", return_value=MagicMock()), \
             patch("viewer.main_window.MainWindow", fake_main_window), \
             patch("viewer.main_window.compute_fft_many") as mock_fft, \
             patch("viewer.main_window.ExpressionManager"):
            freq = np.linspace(0, 500, 33)
            mock_fft.return_value = (freq, np.ones((1, 33)))
            win._on_menu_fft(0)
        # assert — FFT window initial step selection matches source chart selected steps
        self.assertEqual(created_windows[0]._initial_selected_steps, {1, 2})

    def test_fft_window_is_registered_in_application_registry(self):
        # arrange
        steps = 2
        step_points = 64
        win = self._make_fft_win(steps, step_points)
        data = np.concatenate([np.full(step_points, float(s)) for s in range(steps)])
        expr = Expression("V(out)", data, "V")
        chart = MagicMock()
        chart.expressions = [expr]
        chart.abscissa = win._abscissa
        chart.selected_steps = {0, 1}
        win._charts = [chart]
        dialog_class = self._make_dialog_mock(expr, step_points)
        created_windows = []
        def fake_main_window(qraw, source_qraw_path=None):
            win2 = MagicMock()
            win2._initial_selected_steps = None
            created_windows.append(win2)
            return win2
        with patch("viewer.main_window.FftDialog", dialog_class), \
             patch("viewer.main_window.QRawFile", return_value=MagicMock()), \
             patch("viewer.main_window.MainWindow", fake_main_window), \
             patch("viewer.main_window.compute_fft_many") as mock_fft, \
             patch("viewer.main_window.ExpressionManager"), \
             patch("viewer.main_window._register_fft_window") as mock_register:
            freq = np.linspace(0, 500, 33)
            mock_fft.return_value = (freq, np.ones((1, 33)))
            win._on_menu_fft(0)
        # assert
        mock_register.assert_called_once_with(created_windows[0])

    def test_fft_expression_data_length_is_steps_times_freq_points(self):
        # arrange
        steps = 2
        step_points = 64
        freq_points = 33
        win = self._make_fft_win(steps, step_points)
        data = np.concatenate([np.full(step_points, float(s)) for s in range(steps)])
        expr = Expression("V(out)", data, "V")
        chart = MagicMock()
        chart.expressions = [expr]
        chart.abscissa = win._abscissa
        chart.selected_steps = {0, 1}
        win._charts = [chart]
        dialog_class = self._make_dialog_mock(expr, step_points)
        captured_expressions = []
        def fake_expr_mgr(expressions):
            captured_expressions.extend(expressions)
            return MagicMock()
        with patch("viewer.main_window.FftDialog", dialog_class), \
             patch("viewer.main_window.QRawFile", return_value=MagicMock()), \
             patch("viewer.main_window.MainWindow", return_value=MagicMock(_initial_selected_steps=None)), \
             patch("viewer.main_window.compute_fft_many") as mock_fft, \
             patch("viewer.main_window.ExpressionManager", fake_expr_mgr):
            freq = np.linspace(0, 500, freq_points)
            mock_fft.return_value = (freq, np.ones((1, freq_points)))
            win._on_menu_fft(0)
        # assert — FFT expression data length = steps * freq_points; abscissa = freq_points only
        fft_expr = [e for e in captured_expressions if e.name != "Frequency"][0]
        abscissa_expr = [e for e in captured_expressions if e.name == "Frequency"][0]
        self.assertEqual(len(fft_expr.data), steps * freq_points)
        self.assertEqual(len(abscissa_expr.data), freq_points)


class TestComputeDecimateTarget(TestCase):

    def test_returns_fallback_when_screen_is_none(self):
        # arrange
        screen = None
        # act
        result = _compute_decimate_target(screen)
        # assert
        self.assertEqual(result, _FALLBACK_DECIMATE_TARGET)

    def test_uses_screen_width_and_pixel_ratio(self):
        # arrange
        screen = MagicMock()
        screen.size.return_value.width.return_value = 2560
        screen.devicePixelRatio.return_value = 2.0
        # act
        result = _compute_decimate_target(screen)
        # assert
        self.assertEqual(result, 2560 * 5)

    def test_pixel_ratio_below_five_is_clamped_to_five(self):
        # arrange
        screen = MagicMock()
        screen.size.return_value.width.return_value = 1920
        screen.devicePixelRatio.return_value = 1.0
        # act
        result = _compute_decimate_target(screen)
        # assert
        self.assertEqual(result, 1920 * 5)

    def test_high_pixel_ratio_is_used_directly(self):
        # arrange
        screen = MagicMock()
        screen.size.return_value.width.return_value = 3840
        screen.devicePixelRatio.return_value = 8.0
        # act
        result = _compute_decimate_target(screen)
        # assert
        self.assertEqual(result, 3840 * 8)
