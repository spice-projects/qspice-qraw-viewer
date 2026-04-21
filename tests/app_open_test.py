from unittest import TestCase
from unittest.mock import MagicMock, patch

from viewer.app_open import open_qraw_as_window


class TestOpenQrawAsWindow(TestCase):

    def test_open_qraw_as_window_returns_window_on_success(self):
        # arrange
        qraw_file = MagicMock()
        created_window = MagicMock()
        window_factory = MagicMock(return_value=created_window)
        # act
        with patch("viewer.app_open.QRawFile.load", return_value=qraw_file):
            result = open_qraw_as_window("test.qraw", window_factory)
        # assert
        self.assertIs(result, created_window)
        window_factory.assert_called_once_with(qraw_file)

    def test_open_qraw_as_window_returns_none_on_parse_failure(self):
        # arrange
        window_factory = MagicMock()
        # act
        with patch("viewer.app_open.QRawFile.load", return_value=None):
            result = open_qraw_as_window("bad.qraw", window_factory)
        # assert
        self.assertIsNone(result)
        window_factory.assert_not_called()
