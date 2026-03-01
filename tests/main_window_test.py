import sys
from unittest import TestCase
from unittest.mock import MagicMock

# mock PySide6 submodules before importing main_window, which requires Qt at import time
sys.modules.setdefault("PySide6", MagicMock())
sys.modules.setdefault("PySide6.QtCore", MagicMock())
sys.modules.setdefault("PySide6.QtGui", MagicMock())
sys.modules.setdefault("PySide6.QtGraphs", MagicMock())
sys.modules.setdefault("PySide6.QtQml", MagicMock())
sys.modules.setdefault("PySide6.QtQuick", MagicMock())
sys.modules.setdefault("PySide6.QtWidgets", MagicMock())
# Slot must act as a pass-through decorator so @Slot(...) does not replace the method with a mock
sys.modules["PySide6.QtCore"].Slot = lambda *a, **kw: (lambda f: f)
# QMainWindow must be a concrete class so that MainWindow can genuinely inherit from it
sys.modules["PySide6.QtWidgets"].QMainWindow = type("QMainWindow", (), {})

from viewer.main_window import MainWindow  # noqa: E402


class TestMainWindow(TestCase):

    def test_zoom_in_reduces_window(self):
        # arrange
        qraw = MagicMock()
        qraw.variables = [MagicMock(values=list(range(20)))]
        win = MainWindow.__new__(MainWindow)
        win.qraw_file = qraw
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
        qraw = MagicMock()
        qraw.variables = [MagicMock(values=list(range(20)))]
        win = MainWindow.__new__(MainWindow)
        win.qraw_file = qraw
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
        qraw = MagicMock()
        qraw.variables = [MagicMock(values=list(range(20)))]
        win = MainWindow.__new__(MainWindow)
        win.qraw_file = qraw
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
        qraw = MagicMock()
        qraw.variables = [MagicMock(values=list(range(20)))]
        win = MainWindow.__new__(MainWindow)
        win.qraw_file = qraw
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
        qraw = MagicMock()
        qraw.variables = [MagicMock(values=list(range(20)))]
        win = MainWindow.__new__(MainWindow)
        win.qraw_file = qraw
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
        qraw = MagicMock()
        qraw.variables = [MagicMock(values=list(range(20)))]
        win = MainWindow.__new__(MainWindow)
        win.qraw_file = qraw
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
        qraw = MagicMock()
        qraw.variables = [MagicMock(values=list(range(20)))]
        win = MainWindow.__new__(MainWindow)
        win.qraw_file = qraw
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
        qraw = MagicMock()
        qraw.variables = [MagicMock(values=list(range(20)))]
        win = MainWindow.__new__(MainWindow)
        win.qraw_file = qraw
        win._charts = []
        win._abscissa_from_index = 10
        win._abscissa_to_index = 20
        # act — pan right when already at the right boundary must not exceed total length
        win._on_horizontal_zoom(0, 0.2, 1.2, 1.0)
        # assert
        self.assertEqual(win._abscissa_from_index, 10)
        self.assertEqual(win._abscissa_to_index, 20)
