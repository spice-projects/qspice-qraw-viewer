import sys
from unittest import TestCase
from unittest.mock import MagicMock, patch, ANY

import numpy as np

# mock PySide6 submodules before importing smith_chart_window, which requires Qt at import time
sys.modules.setdefault("PySide6", MagicMock())
sys.modules.setdefault("PySide6.QtCore", MagicMock())
sys.modules.setdefault("PySide6.QtGui", MagicMock())
sys.modules.setdefault("PySide6.QtQml", MagicMock())
sys.modules.setdefault("PySide6.QtQuick", MagicMock())
sys.modules.setdefault("PySide6.QtWidgets", MagicMock())

# Slot must act as a pass-through decorator so @Slot(...) does not replace the method with a mock
sys.modules["PySide6.QtCore"].Slot = lambda *a, **kw: (lambda f: f)

# Mock QmlElement before it is used as a decorator
def mock_qml_element(cls):
    return cls
sys.modules["PySide6.QtQml"].QmlElement = mock_qml_element

# QMainWindow and QQuickItem must be concrete classes
sys.modules["PySide6.QtWidgets"].QMainWindow = type("QMainWindow", (), {})

# Mock UpdatePaintNodeData as a dummy type
class MockUpdatePaintNodeData:
    pass

# Mock QQuickItem with needed attributes
class MockQQuickItem:
    UpdatePaintNodeData = MockUpdatePaintNodeData
    class Flag:
        ItemHasContents = 0x1
    def __init__(self, *args, **kwargs): pass
    def setFlag(self, *args, **kwargs): pass
    def width(self): return 1000
    def height(self): return 1000
    def update(self): pass
    def window(self): return MagicMock()

sys.modules["PySide6.QtQuick"].QQuickItem = MockQQuickItem
sys.modules["PySide6.QtQuick"].QSGNode = type("QSGNode", (), {})

import importlib  # noqa: E402
import viewer.smith_chart_window  # noqa: E402
importlib.reload(viewer.smith_chart_window)
from viewer.smith_chart_window import SmithChartWindow, SmithTraceItem  # noqa: E402

from viewer.expression import Expression  # noqa: E402
from viewer.qraw_file import QRawFile, StepInformation  # noqa: E402


class TestSmithChartWindow(TestCase):

    def _make_win(self):
        # build a mock QRawFile with basic metadata
        qraw_file = MagicMock(spec=QRawFile)
        qraw_file.filename = MagicMock()
        qraw_file.filename.name = "test.qraw"
        qraw_file.chart_type = "AC"
        qraw_file.abscissa = MagicMock()
        qraw_file.abscissa_scale = MagicMock()
        qraw_file.expression_manager = MagicMock()
        qraw_file.step_information = StepInformation([], [()], [slice(0, 100)], [(0.0, 1.0)])
        qraw_file.get_plot_suggestions.return_value = []
        
        # bypass __init__ to avoid QML view instantiation and styling
        win = SmithChartWindow.__new__(SmithChartWindow)
        win._qraw_file = qraw_file
        win._abscissa = qraw_file.abscissa
        win._abscissa_scale = qraw_file.abscissa_scale
        win._expression_manager = qraw_file.expression_manager
        win._step_information = qraw_file.step_information
        win._selected_steps = {0}
        win._expressions = {}
        win._next_color_index = 0
        win._root = MagicMock()
        
        return win

    def test_add_plots_new_expression_assigns_color_and_renders(self):
        # arrange
        win = self._make_win()
        trace_item = MagicMock()
        win._root.findChild.return_value = trace_item
        
        expr = Expression("S11", np.array([0.5 + 0.5j] * 100), "S")
        
        # act
        with patch("viewer.smith_chart_window.QMetaObject.invokeMethod") as mock_invoke:
            win._add_plots({expr})
            
        # assert — color assigned and trace cached
        self.assertIn(expr, win._expressions)
        color, rendered_traces = win._expressions[expr]
        self.assertIsNotNone(color)
        self.assertIn(0, rendered_traces)
        
        # assert — invokeMethod called to update UI
        mock_invoke.assert_called_once()
        args = mock_invoke.call_args[0]
        self.assertEqual(args[1], "plot")
        traces = args[2] # In QMetaObject.invokeMethod(item, "method", Q_ARG(...))
        # But our code uses: QMetaObject.invokeMethod(trace_item, "plot", Q_ARG("QVariant", traces_to_render))
        # The mock capture might vary depending on how Q_ARG is mocked.
        # It's actually: invokeMethod(target, method_name, *args)
        # So: target=trace_item, method_name="plot", arg3=Q_ARG(...)
        self.assertEqual(args[0], trace_item)
        self.assertEqual(args[1], "plot")

    def test_add_plots_removes_unselected_expressions(self):
        # arrange
        win = self._make_win()
        expr1 = Expression("S11", np.array([0.1j] * 10), "S")
        expr2 = Expression("S22", np.array([0.2j] * 10), "S")
        win._expressions = {expr1: (MagicMock(), {0: np.zeros((10, 2))})}
        
        # act — request expr2 only
        win._add_plots({expr2})
        
        # assert — expr1 removed, expr2 added
        self.assertNotIn(expr1, win._expressions)
        self.assertIn(expr2, win._expressions)

    def test_add_plots_incremental_step_update(self):
        # arrange — simulate 2 steps, step 0 already rendered
        win = self._make_win()
        win._step_information = StepInformation([], [(), ()], [slice(0, 10), slice(10, 20)], [(0.0, 1.0), (1.0, 2.0)])
        expr = Expression("S11", np.array([0.1j] * 20), "S")
        existing_gamma = np.zeros((10, 2))
        win._expressions = {expr: (MagicMock(), {0: existing_gamma})}
        
        # user selects steps 0 and 1
        win._selected_steps = {0, 1}
        
        # act
        win._add_plots({expr})
        
        # assert — existing step 0 trace preserved, new step 1 trace added
        _, rendered_traces = win._expressions[expr]
        self.assertIs(rendered_traces[0], existing_gamma)
        self.assertIn(1, rendered_traces)
        self.assertEqual(rendered_traces[1].shape, (10, 2))

    def test_add_plots_filters_non_finite_values(self):
        # arrange
        win = self._make_win()
        trace_item = MagicMock()
        win._root.findChild.return_value = trace_item
        # data with NaN and Inf
        data = np.array([0.5+0.5j, np.nan, 0.1+0.1j, np.inf])
        expr = Expression("S11", data, "S")
        win._step_information = StepInformation([], [()], [slice(0, 4)], [(0.0, 1.0)])
        
        # act
        win._add_plots({expr})
        
        # assert — only the 2 finite points remain in the rendered trace
        _, rendered_traces = win._expressions[expr]
        gamma = rendered_traces[0]
        self.assertEqual(len(gamma), 2)
        np.testing.assert_array_equal(gamma, [[0.5, 0.5], [0.1, 0.1]])

    def test_clear_resets_state_and_ui(self):
        # arrange
        win = self._make_win()
        win._expressions = {MagicMock(): (MagicMock(), {})}
        win._next_color_index = 5
        trace_item = MagicMock()
        win._root.findChild.return_value = trace_item
        
        # act
        with patch("viewer.smith_chart_window.QMetaObject.invokeMethod") as mock_invoke:
            win._clear()
            
        # assert — state reset
        self.assertEqual(win._expressions, {})
        self.assertEqual(win._next_color_index, 0)
        # assert — UI cleared
        mock_invoke.assert_called_once_with(trace_item, "plot", ANY)

    def test_on_menu_step_tool_refreshes_plots_on_accept(self):
        # arrange
        win = self._make_win()
        win._add_plots = MagicMock()
        expr = Expression("S11", np.zeros(10), "S")
        win._expressions = {expr: (MagicMock(), {})}
        
        dialog = MagicMock()
        dialog.exec.return_value = MagicMock()
        dialog.exec.return_value.Accepted = True # simplification for mock behavior
        dialog.selected_steps = {1, 2}
        
        # act
        with patch("viewer.smith_chart_window.StepToolDialog", return_value=dialog):
            # make Accepted match the return value to trigger logic
            with patch("viewer.smith_chart_window.StepToolDialog.DialogCode.Accepted", dialog.exec.return_value):
                win._on_menu_step_tool()
                
        # assert — steps updated and plots refreshed
        self.assertEqual(win._selected_steps, {1, 2})
        win._add_plots.assert_called_once()
        passed_exprs = win._add_plots.call_args[0][0]
        self.assertIn(expr, passed_exprs)


class TestSmithTraceItem(TestCase):

    def test_get_pixel_pts_applies_decimation(self):
        # arrange
        item = SmithTraceItem()
        # 10000 points on the unit disk (Gamma=1)
        theta = np.linspace(0, 2*np.pi, 10000)
        pts_gamma = np.column_stack((np.cos(theta), np.sin(theta)))
        cx, cy, R = 500, 500, 450
        
        # act — _get_pixel_pts(self, pts_gamma, cx, cy, R)
        with patch("viewer.smith_chart_window.decimate_xy", side_effect=lambda x, y, **kw: (x[::10], y[::10])) as mock_decimate:
            px = item._get_pixel_pts(pts_gamma, cx, cy, R)
            
        # assert — decimation called and output points are reduced
        mock_decimate.assert_called_once()
        self.assertEqual(len(px), 1000)
        # check first point mapping: cx + cos(0)*R = 500 + 450 = 950
        self.assertAlmostEqual(px[0, 0], 950.0)

    def test_plot_sets_dirty_and_triggers_update(self):
        # arrange
        item = SmithTraceItem()
        item.update = MagicMock()
        traces = [(np.zeros((10, 2)), "test", MagicMock())]
        
        # act
        item.plot(traces)
        
        # assert
        self.assertEqual(item._traces, traces)
        self.assertTrue(item._dirty)
        item.update.assert_called_once()
