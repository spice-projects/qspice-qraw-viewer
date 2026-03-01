from unittest import TestCase, mock

from viewer.main_window import MainWindow


def _make_window():
    qraw = mock.Mock()
    qraw.abscissa_points = 20
    qraw.variables = [mock.Mock(values=list(range(20)))]
    qraw.abscissa_scale = mock.Mock(value=mock.Mock(unit=""))
    qraw.filename = mock.Mock(name="dummy.qraw")
    win = MainWindow.__new__(MainWindow)
    win.qraw_file = qraw
    win._charts = []
    win._abscissa_from_index = 0
    win._abscissa_to_index = 20
    return win


class TestMainWindow(TestCase):

    def test_zoom_in_reduces_window(self):
        # arrange
        win = _make_window()
        win._abscissa_from_index = 0
        win._abscissa_to_index = 20
        # act
        win._on_horizontal_zoom(0, 0.3, 0.7, zoom_factor=0.5)
        # assert
        new_width = win._abscissa_to_index - win._abscissa_from_index
        self.assertLess(new_width, 20)
        self.assertGreaterEqual(win._abscissa_from_index, 0)
        self.assertLessEqual(win._abscissa_to_index, 20)

    def test_zoom_out_moves_window_outward(self):
        # arrange
        win = _make_window()
        win._abscissa_from_index = 5
        win._abscissa_to_index = 15
        old_width = win._abscissa_to_index - win._abscissa_from_index
        # act
        win._on_horizontal_zoom(0, 0.25, 0.75, zoom_factor=2.0)
        # assert
        new_width = win._abscissa_to_index - win._abscissa_from_index
        self.assertNotEqual(new_width, old_width)
        self.assertGreaterEqual(new_width, 2)
        self.assertLessEqual(win._abscissa_to_index, 20)

    def test_zoom_out_at_boundary_saturates(self):
        # arrange
        win = _make_window()
        win._abscissa_from_index = 0
        win._abscissa_to_index = 20
        # act
        for _ in range(5):
            win._on_horizontal_zoom(0, 0.0, 1.0, zoom_factor=2.0)
        # assert
        self.assertEqual(win._abscissa_from_index, 0)
        self.assertEqual(win._abscissa_to_index, 20)

    def test_minimum_window_enforced(self):
        # arrange
        win = _make_window()
        win._abscissa_from_index = 0
        win._abscissa_to_index = 20
        # act
        for _ in range(50):
            win._on_horizontal_zoom(0, 0.4, 0.6, zoom_factor=0.5)
        # assert
        width = win._abscissa_to_index - win._abscissa_from_index
        self.assertGreaterEqual(width, 2)

    def test_pan_moves_window_right(self):
        # arrange
        win = _make_window()
        win._abscissa_from_index = 5
        win._abscissa_to_index = 15
        # act
        win._on_horizontal_zoom(0, 0.2, 1.2, zoom_factor=1.0)
        # assert
        self.assertGreater(win._abscissa_from_index, 5)
        self.assertGreater(win._abscissa_to_index, 15)

    def test_pan_moves_window_left(self):
        # arrange
        win = _make_window()
        win._abscissa_from_index = 5
        win._abscissa_to_index = 15
        # act
        win._on_horizontal_zoom(0, -0.2, 0.8, zoom_factor=1.0)
        # assert
        self.assertLess(win._abscissa_from_index, 5)
        self.assertLess(win._abscissa_to_index, 15)

    def test_pan_clamps_at_left_boundary(self):
        # arrange
        win = _make_window()
        win._abscissa_from_index = 0
        win._abscissa_to_index = 10
        # act
        win._on_horizontal_zoom(0, -0.2, 0.8, zoom_factor=1.0)
        # assert
        self.assertEqual(win._abscissa_from_index, 0)
        self.assertEqual(win._abscissa_to_index, 10)

    def test_pan_clamps_at_right_boundary(self):
        # arrange
        win = _make_window()
        total = win.qraw_file.abscissa_points
        win._abscissa_from_index = total - 10
        win._abscissa_to_index = total
        # act
        win._on_horizontal_zoom(0, 0.2, 1.2, zoom_factor=1.0)
        # assert
        self.assertEqual(win._abscissa_from_index, total - 10)
        self.assertEqual(win._abscissa_to_index, total)
