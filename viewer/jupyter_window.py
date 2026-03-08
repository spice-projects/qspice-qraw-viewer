import json
import logging
import re
import socket
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMainWindow

from .notebook_template import NOTEBOOKS_DIR, ensure_starter_notebook

logger = logging.getLogger(__name__)

# background color to match the rest of the application
_BG = "#1a1b1e"

# interval between server-readiness polls in milliseconds
_POLL_INTERVAL_MS = 500

# maximum total wait time for the server to start in milliseconds
_START_TIMEOUT_MS = 30_000

# JupyterLab user-settings directory placed beside the notebooks folder
_LAB_SETTINGS_DIR = NOTEBOOKS_DIR.parent / "lab-settings"

# relative path of the theme settings file inside the user-settings directory
_THEME_SETTINGS_REL = "@jupyterlab/apputils-extension/themes.jupyterlab-settings"


def _find_free_port() -> int:
    # bind to port 0 to let the OS assign a free ephemeral port, then release the socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_lab_settings() -> None:
    # create the full directory hierarchy for the theme settings file
    theme_file = _LAB_SETTINGS_DIR / _THEME_SETTINGS_REL
    theme_file.parent.mkdir(parents=True, exist_ok=True)
    # always write the dark theme preference so it is never overridden by a stale user config
    theme_file.write_text(json.dumps({"theme": "JupyterLab Dark"}, indent=1), encoding="utf-8")


class JupyterWindow(QMainWindow):
    # emitted once JupyterLab has fully loaded and the window is visible
    ready = Signal()

    def __init__(self, qraw_path: Path):
        super().__init__()
        # store the path to the simulation file
        self._qraw_path = qraw_path
        # pick an unused port for this session
        self._port = _find_free_port()
        # authentication token extracted from the Jupyter process stdout
        self._token: str | None = None
        # cumulative milliseconds polled so far, used to detect startup timeout
        self._elapsed_ms = 0
        # ensure the starter notebook file exists, reuse if already present
        self._notebook_path = ensure_starter_notebook(qraw_path)
        # write the dark-theme settings file before launching the server
        _ensure_lab_settings()
        # configure window chrome
        self.setWindowTitle(f"Jupyter — {qraw_path.name}")
        self.resize(1400, 900)
        # web view with dark page background — prevents a white flash when the page loads
        self._web_view = QWebEngineView()
        self._web_view.page().setBackgroundColor(QColor(_BG))
        self._web_view.loadFinished.connect(self._on_load_finished)
        self.setCentralWidget(self._web_view)
        # window stays hidden until _on_load_finished — caller must not call show()
        # create the managed subprocess for jupyter lab
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_process_finished)
        # timer that polls for TCP reachability once the token is known
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._on_poll)
        # launch the server
        self._start_jupyter()

    def _start_jupyter(self) -> None:
        # resolve the jupyter executable from the same venv as the running interpreter
        jupyter_exe = str(Path(sys.executable).parent / "jupyter")
        # notebook-dir is set to the user home so the file browser covers the full home tree
        args = [
            "lab",
            "--no-browser",
            f"--port={self._port}",
            f"--notebook-dir={Path.home()}",
            "--ServerApp.open_browser=False",
            "--ContentsManager.allow_hidden=True",
        ]
        # inject settings dir via environment variable — avoids invalid CLI flag warnings
        env = QProcessEnvironment.systemEnvironment()
        env.insert("JUPYTERLAB_SETTINGS_DIR", str(_LAB_SETTINGS_DIR))
        self._process.setProcessEnvironment(env)
        logger.info("Starting JupyterLab: %s %s", jupyter_exe, " ".join(args))
        self._process.start(jupyter_exe, args)

    @Slot()
    def _on_output(self) -> None:
        # decode all bytes buffered since the last signal emission
        raw = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        logger.debug("Jupyter stdout: %s", raw.rstrip())
        # scan for the token URL only once — it appears on the first startup message
        if self._token is not None:
            return
        # extract the token from the URL using a regex — this is the only way to get the token without parsing the HTML page or using an API
        match = re.search(r"http://(?:localhost|127\.0\.0\.1):\d+/lab\?token=([a-f0-9]+)", raw)
        if match:
            # token
            self._token = match.group(1)
            # log information
            logger.info("JupyterLab token acquired, polling for readiness on port %d", self._port)
            # begin polling now that the token is known
            self._poll_timer.start()

    @Slot()
    def _on_poll(self) -> None:
        # advance the timeout counter
        self._elapsed_ms += _POLL_INTERVAL_MS
        if self._elapsed_ms >= _START_TIMEOUT_MS:
            # timeout expired — stop polling and log an error; the user can still open the page manually using the token printed in the logs if the server eventually starts
            self._poll_timer.stop()
            # log information
            logger.error("Timed out waiting for JupyterLab to start on port %d", self._port)
            # exit
            return
        # a successful TCP connection confirms the HTTP server is up
        try:
            with socket.create_connection(("127.0.0.1", self._port), timeout=0.2):
                pass
        except OSError:
            # server not yet accepting connections — wait for the next tick
            return
        # server is ready — begin loading the page; the stack stays on the loading widget
        self._poll_timer.stop()
        self._load_web_view()

    def _load_web_view(self) -> None:
        # relative path from the home directory determines the /lab/tree/ URL segment
        notebook_rel = self._notebook_path.relative_to(Path.home()).as_posix()
        url = f"http://127.0.0.1:{self._port}/lab/tree/{notebook_rel}?token={self._token}"
        logger.info("Loading JupyterLab at: %s", url)
        # start the page load — the switch to the web view happens in _on_load_finished
        self._web_view.load(QUrl(url))

    @Slot(bool)
    def _on_load_finished(self, ok: bool) -> None:
        # show the window only once JupyterLab has finished rendering — no visible loading state
        if ok:
            self.show()
            self.ready.emit()

    @Slot(int, QProcess.ExitStatus)
    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        logger.info("JupyterLab process exited with code %d", exit_code)

    def closeEvent(self, event) -> None:
        # stop polling before tearing down the process
        self._poll_timer.stop()
        # terminate the server gracefully, then force-kill if it does not exit in time
        if self._process.state() != QProcess.ProcessState.NotRunning:
            logger.info("Terminating JupyterLab process on port %d", self._port)
            self._process.terminate()
            if not self._process.waitForFinished(3000):
                self._process.kill()
        super().closeEvent(event)
