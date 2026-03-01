import unittest
from unittest import mock

from viewer.main_window import MainWindow


class TestMainWindowZoom(unittest.TestCase):
    def setUp(self):
        # create a mock qraw file with the minimal attributes used by the
        # constructor and the zoom handler
        self.qraw = mock.Mock()
        self.qraw.abscissa_points = 20
        self.qraw.variables = [mock.Mock(values=list(range(20)))]
        self.qraw.abscissa_scale = mock.Mock(value=mock.Mock(unit=""))
        self.qraw.filename = mock.Mock(name="dummy.qraw")
        # instantiate MainWindow without running its __init__
        self.win = MainWindow.__new__(MainWindow)
        self.win.qraw_file = self.qraw
        self.win._charts = []
        self.win._abscissa_from_index = 0
        self.win._abscissa_to_index = 20

    def test_zoom_in_reduces_window(self):
        # arrange: start with full range
        self.win._abscissa_from_index = 0
        self.win._abscissa_to_index = 20
        # act: zoom in around centre (ratio span 0.4) with factor < 1
        self.win._on_horizontal_zoom(0, 0.3, 0.7, zoom_factor=0.5)
        # assert: window should be smaller than 20 and remain within bounds
        new_width = self.win._abscissa_to_index - self.win._abscissa_from_index
        self.assertLess(new_width, 20)
        self.assertGreaterEqual(self.win._abscissa_from_index, 0)
        self.assertLessEqual(self.win._abscissa_to_index, 20)

    def test_zoom_out_moves_window_outward(self):
        # arrange: start with a centred subrange
        self.win._abscissa_from_index = 5
        self.win._abscissa_to_index = 15
        old_width = self.win._abscissa_to_index - self.win._abscissa_from_index
        # act: request zoom out with factor >1
        self.win._on_horizontal_zoom(0, 0.25, 0.75, zoom_factor=2.0)
        # assert: at least one edge moved outward (left decreased or right increased)
        # the window should change (either expand or translate)
        new_width = self.win._abscissa_to_index - self.win._abscissa_from_index
        self.assertNotEqual(new_width, old_width)
        # window width must still be >= minimum and within bounds
        self.assertGreaterEqual(new_width, 2)
        self.assertLessEqual(self.win._abscissa_to_index, 20)

    def test_zoom_out_at_boundary_saturates(self):
        # arrange: already showing full range
        self.win._abscissa_from_index = 0
        self.win._abscissa_to_index = 20
        # act: zoom out many times
        for _ in range(5):
            self.win._on_horizontal_zoom(0, 0.0, 1.0, zoom_factor=2.0)
        # assert: still full range
        self.assertEqual(self.win._abscissa_from_index, 0)
        self.assertEqual(self.win._abscissa_to_index, 20)

    def test_minimum_window_enforced(self):
        # arrange: zoom repeatedly in until we hit the minimum size
        self.win._abscissa_from_index = 0
        self.win._abscissa_to_index = 20
        # perform many zoom-in steps
        for _ in range(50):
            self.win._on_horizontal_zoom(0, 0.4, 0.6, zoom_factor=0.5)
        # width should not fall below 2
        width = self.win._abscissa_to_index - self.win._abscissa_from_index
        self.assertGreaterEqual(width, 2)

    def test_pan_moves_window_right(self):
        # arrange: centered window such that pan can move right
        self.win._abscissa_from_index = 5
        self.win._abscissa_to_index = 15
        # pan right: positive left_ratio (span==1 triggers pan logic)
        self.win._on_horizontal_zoom(0, 0.2, 1.2, zoom_factor=1.0)
        # new_from should be greater than previous start
        self.assertGreater(self.win._abscissa_from_index, 5)
        self.assertGreater(self.win._abscissa_to_index, 15)

    def test_pan_moves_window_left(self):
        # arrange: window away from left edge
        self.win._abscissa_from_index = 5
        self.win._abscissa_to_index = 15
        # pan left: negative left_ratio
        self.win._on_horizontal_zoom(0, -0.2, 0.8, zoom_factor=1.0)
        self.assertLess(self.win._abscissa_from_index, 5)
        self.assertLess(self.win._abscissa_to_index, 15)

    def test_pan_clamps_at_left_boundary(self):
        # arrange at left boundary
        self.win._abscissa_from_index = 0
        self.win._abscissa_to_index = 10
        # request pan left
        self.win._on_horizontal_zoom(0, -0.2, 0.8, zoom_factor=1.0)
        self.assertEqual(self.win._abscissa_from_index, 0)
        self.assertEqual(self.win._abscissa_to_index, 10)

    def test_pan_clamps_at_right_boundary(self):
        # arrange near right edge
        total = self.qraw.abscissa_points
        self.win._abscissa_from_index = total - 10
        self.win._abscissa_to_index = total
        # request pan right
        self.win._on_horizontal_zoom(0, 0.2, 1.2, zoom_factor=1.0)
        self.assertEqual(self.win._abscissa_from_index, total - 10)
        self.assertEqual(self.win._abscissa_to_index, total)
