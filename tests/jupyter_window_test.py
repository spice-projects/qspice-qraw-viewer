import json
import socket
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

# mock PySide6 submodules before importing jupyter_window, which requires Qt at import time
sys.modules.setdefault("PySide6", MagicMock())
sys.modules.setdefault("PySide6.QtCore", MagicMock())
sys.modules.setdefault("PySide6.QtGui", MagicMock())
sys.modules.setdefault("PySide6.QtWebEngineWidgets", MagicMock())
sys.modules.setdefault("PySide6.QtWidgets", MagicMock())
# Slot must act as a pass-through decorator so @Slot(...) does not replace the method with a mock
sys.modules["PySide6.QtCore"].Slot = lambda *a, **kw: (lambda f: f)
# Signal must be callable at class-definition time (used as a class attribute)
sys.modules["PySide6.QtCore"].Signal = MagicMock()
# QMainWindow must be a concrete class so that JupyterWindow can genuinely inherit from it
sys.modules["PySide6.QtWidgets"].QMainWindow = type("QMainWindow", (), {})

from viewer.jupyter_window import (  # noqa: E402
    JupyterWindow,
    _POLL_INTERVAL_MS,
    _START_TIMEOUT_MS,
    _ensure_lab_settings,
    _find_free_port,
)


class TestFindFreePort(TestCase):

    def test_returns_integer(self):
        # arrange — no preconditions required
        # act
        port = _find_free_port()
        # assert
        self.assertIsInstance(port, int)

    def test_port_is_in_valid_range(self):
        # arrange — no preconditions required
        # act
        port = _find_free_port()
        # assert
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)

    def test_returned_port_is_bindable(self):
        # arrange — no preconditions required
        # act
        port = _find_free_port()
        # assert — if _find_free_port released the socket, binding to it should succeed
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))


class TestEnsureLabSettings(TestCase):

    def test_creates_theme_settings_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange
            lab_settings_dir = Path(tmpdir) / "lab-settings"
            # act
            with patch("viewer.jupyter_window._LAB_SETTINGS_DIR", lab_settings_dir):
                _ensure_lab_settings()
            # assert — expected file must exist somewhere inside the settings dir
            theme_file = lab_settings_dir / "@jupyterlab/apputils-extension/themes.jupyterlab-settings"
            self.assertTrue(theme_file.exists())

    def test_theme_is_dark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange
            lab_settings_dir = Path(tmpdir) / "lab-settings"
            # act
            with patch("viewer.jupyter_window._LAB_SETTINGS_DIR", lab_settings_dir):
                _ensure_lab_settings()
            # assert
            theme_file = lab_settings_dir / "@jupyterlab/apputils-extension/themes.jupyterlab-settings"
            content = json.loads(theme_file.read_text(encoding="utf-8"))
            self.assertEqual(content["theme"], "JupyterLab Dark")

    def test_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # arrange — pre-write a stale settings file with the wrong theme
            lab_settings_dir = Path(tmpdir) / "lab-settings"
            theme_file = lab_settings_dir / "@jupyterlab/apputils-extension/themes.jupyterlab-settings"
            theme_file.parent.mkdir(parents=True)
            theme_file.write_text(json.dumps({"theme": "JupyterLab Light"}), encoding="utf-8")
            # act
            with patch("viewer.jupyter_window._LAB_SETTINGS_DIR", lab_settings_dir):
                _ensure_lab_settings()
            # assert — stale content must be replaced with the dark theme
            content = json.loads(theme_file.read_text(encoding="utf-8"))
            self.assertEqual(content["theme"], "JupyterLab Dark")


class TestJupyterWindowOnOutput(TestCase):

    def test_extracts_token_from_stdout(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._token = None
        win._port = 8888
        win._poll_timer = MagicMock()
        win._process = MagicMock()
        url_line = b"http://127.0.0.1:8888/lab?token=abc123def456abc1\n"
        win._process.readAllStandardOutput.return_value.data.return_value = url_line
        # act
        win._on_output()
        # assert
        self.assertEqual(win._token, "abc123def456abc1")

    def test_starts_poll_timer_when_token_found(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._token = None
        win._port = 8888
        win._poll_timer = MagicMock()
        win._process = MagicMock()
        url_line = b"http://localhost:8888/lab?token=deadbeef12345678\n"
        win._process.readAllStandardOutput.return_value.data.return_value = url_line
        # act
        win._on_output()
        # assert
        win._poll_timer.start.assert_called_once()

    def test_does_not_overwrite_token_on_second_call(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._token = "firsttoken1234567"
        win._poll_timer = MagicMock()
        win._process = MagicMock()
        url_line = b"http://127.0.0.1:8888/lab?token=secondtoken5678\n"
        win._process.readAllStandardOutput.return_value.data.return_value = url_line
        # act
        win._on_output()
        # assert — original token must be preserved
        self.assertEqual(win._token, "firsttoken1234567")

    def test_does_not_start_timer_when_no_token_in_output(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._token = None
        win._poll_timer = MagicMock()
        win._process = MagicMock()
        win._process.readAllStandardOutput.return_value.data.return_value = b"Jupyter is starting...\n"
        # act
        win._on_output()
        # assert
        win._poll_timer.start.assert_not_called()


class TestJupyterWindowOnPoll(TestCase):

    def test_stops_timer_when_timeout_reached(self):
        # arrange — set elapsed to exactly one interval before the timeout limit
        win = JupyterWindow.__new__(JupyterWindow)
        win._elapsed_ms = _START_TIMEOUT_MS - _POLL_INTERVAL_MS
        win._port = 19999
        win._poll_timer = MagicMock()
        # act — this tick pushes elapsed to _START_TIMEOUT_MS
        with patch("viewer.jupyter_window.socket.create_connection", side_effect=OSError):
            win._on_poll()
        # assert
        win._poll_timer.stop.assert_called_once()

    def test_increments_elapsed_on_each_call(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._elapsed_ms = 0
        win._port = 19999
        win._poll_timer = MagicMock()
        # act
        with patch("viewer.jupyter_window.socket.create_connection", side_effect=OSError):
            win._on_poll()
        # assert
        self.assertEqual(win._elapsed_ms, _POLL_INTERVAL_MS)

    def test_does_not_stop_timer_when_server_not_ready(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._elapsed_ms = 0
        win._port = 19999
        win._poll_timer = MagicMock()
        # act — connection refused means server is not yet up
        with patch("viewer.jupyter_window.socket.create_connection", side_effect=OSError):
            win._on_poll()
        # assert — timer must keep running
        win._poll_timer.stop.assert_not_called()


class TestJupyterWindowLoadWebView(TestCase):

    def test_url_contains_port(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._port = 54321
        win._token = "mytoken1234567890"
        win._notebook_path = Path.home() / ".qraw-viewer/notebooks/test.ipynb"
        win._web_view = MagicMock()
        # act
        with patch("viewer.jupyter_window.QUrl") as mock_qurl:
            win._load_web_view()
            url_arg = mock_qurl.call_args[0][0]
        # assert
        self.assertIn(":54321/", url_arg)

    def test_url_contains_token(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._port = 54321
        win._token = "mytoken1234567890"
        win._notebook_path = Path.home() / ".qraw-viewer/notebooks/test.ipynb"
        win._web_view = MagicMock()
        # act
        with patch("viewer.jupyter_window.QUrl") as mock_qurl:
            win._load_web_view()
            url_arg = mock_qurl.call_args[0][0]
        # assert
        self.assertIn("token=mytoken1234567890", url_arg)

    def test_url_targets_localhost(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._port = 54321
        win._token = "mytoken1234567890"
        win._notebook_path = Path.home() / ".qraw-viewer/notebooks/test.ipynb"
        win._web_view = MagicMock()
        # act
        with patch("viewer.jupyter_window.QUrl") as mock_qurl:
            win._load_web_view()
            url_arg = mock_qurl.call_args[0][0]
        # assert — must bind to loopback to prevent network exposure
        self.assertTrue(url_arg.startswith("http://127.0.0.1:"))

    def test_url_contains_notebook_path_segment(self):
        # arrange
        win = JupyterWindow.__new__(JupyterWindow)
        win._port = 54321
        win._token = "mytoken1234567890"
        win._notebook_path = Path.home() / ".qraw-viewer/notebooks/my_sim.ipynb"
        win._web_view = MagicMock()
        # act
        with patch("viewer.jupyter_window.QUrl") as mock_qurl:
            win._load_web_view()
            url_arg = mock_qurl.call_args[0][0]
        # assert — notebook stem must appear in the /lab/tree/ path
        self.assertIn("my_sim.ipynb", url_arg)
