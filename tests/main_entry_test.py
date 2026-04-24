import importlib
import sys
from unittest import TestCase
from unittest.mock import MagicMock, patch

# mock PySide6 submodules before importing viewer.__main__, which requires Qt at import time
sys.modules.setdefault("PySide6", MagicMock())
sys.modules.setdefault("PySide6.QtCore", MagicMock())
sys.modules.setdefault("PySide6.QtWidgets", MagicMock())


class TestMainEntry(TestCase):

    def test_main_gui_uses_shared_open_path(self):
        # arrange
        main_entry = importlib.import_module("viewer.__main__")
        app = MagicMock()
        app.exec.return_value = 0
        icon = MagicMock()
        icon.isNull.return_value = True
        window = MagicMock()
        # act
        with patch.object(sys, "argv", ["viewer", "test.qraw"]), \
             patch.object(main_entry, "QApplication", return_value=app), \
             patch.object(main_entry, "load_app_icon", return_value=icon), \
             patch.object(main_entry, "open_qraw_as_window", return_value=window) as mock_open:
            with self.assertRaises(SystemExit) as exit_context:
                main_entry.main()
        # assert
        self.assertEqual(exit_context.exception.code, 0)
        mock_open.assert_called_once()
        window.show.assert_called_once_with()
        app.exec.assert_called_once_with()

    def test_main_gui_exits_with_error_when_open_fails(self):
        # arrange
        main_entry = importlib.import_module("viewer.__main__")
        app = MagicMock()
        icon = MagicMock()
        icon.isNull.return_value = True
        # act
        with patch.object(sys, "argv", ["viewer", "test.qraw"]), \
             patch.object(main_entry, "QApplication", return_value=app), \
             patch.object(main_entry, "load_app_icon", return_value=icon), \
             patch.object(main_entry, "open_qraw_as_window", return_value=None) as mock_open:
            with self.assertRaises(SystemExit) as exit_context:
                main_entry.main()
        # assert
        self.assertEqual(exit_context.exception.code, 1)
        mock_open.assert_called_once()
        app.exec.assert_not_called()
