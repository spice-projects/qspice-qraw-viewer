import logging
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, QTimer, QUrl, Slot
from PySide6.QtGui import QAction, QColor, QGuiApplication, QIcon, QKeySequence
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QFileDialog, QMainWindow, QWidget

from .add_plot_dialog import AddPlotDialog
from .app_open import open_qraw_as_window
from .chart import Chart
from .expression import Expression
from .expression_manager import ExpressionManager
from .fft import FftOutput, compute_fft_many
from .fft_dialog import FftDialog
from .jupyter_window import JupyterWindow
from .qraw_file import AbscissaScale, QRawFile
from .step_tool_dialog import StepToolDialog

logger = logging.getLogger(__name__)

_QML_FILE = Path(__file__).parent / "main_window.qml"
_RESOURCE_DIR = Path(__file__).parent / "resources"
_ICON_PATH = _RESOURCE_DIR / "waveform-window-icon.ico"


def load_app_icon() -> QIcon:
    return QIcon(str(_ICON_PATH))


# background color matching the chart dark theme
_BG = "#1a1b1e"

# minimum interval between status-bar updates (≈30 fps)
_MIN_STATUS_INTERVAL = 1.0 / 30

# application-level registry keeps child windows alive independently
# of the source main window that created them
_CHILD_WINDOWS: set[QMainWindow] = set()

# application-level open-file dialog lock prevents stacked dialogs
_OPEN_FILE_DIALOG_ACTIVE = False


def _register_child_window(window: QMainWindow) -> None:
    _CHILD_WINDOWS.add(window)


def _unregister_child_window(window: QMainWindow) -> None:
    _CHILD_WINDOWS.discard(window)


def _format_value(value: float, unit: str) -> str:
    """Format a numeric value with SI prefix and unit, mirroring the QML applyUnit function."""
    # absolute value for prefix selection
    abs_val = abs(value)
    # giga
    if abs_val >= 1e9:
        return f"{value / 1e9:.2f} G{unit}"
    # mega
    if abs_val >= 1e6:
        return f"{value / 1e6:.2f} M{unit}"
    # kilo
    if abs_val >= 1e3:
        return f"{value / 1e3:.2f} k{unit}"
    # base unit
    if abs_val >= 1.0:
        return f"{value:.2f} {unit}"
    # zero
    if abs_val < 1e-15:
        return f"0 {unit}"
    # femto
    if abs_val < 1e-12:
        return f"{value * 1e15:.2f} f{unit}"
    # pico
    if abs_val < 1e-9:
        return f"{value * 1e12:.2f} p{unit}"
    # nano
    if abs_val < 1e-6:
        return f"{value * 1e9:.2f} n{unit}"
    # micro
    if abs_val < 1e-3:
        return f"{value * 1e6:.2f} µ{unit}"
    # milli
    return f"{value * 1e3:.2f} m{unit}"


def _format_values(name: str, values: list[float], unit: str) -> str:
    # check a single value is available
    if len(values) == 1:
        return f"{name} = {_format_value(values[0], unit)}"
    # multiple values: format each and join with commas
    formatted_values = ", ".join(_format_value(v, unit) for v in values)
    # exit
    return f"{name} = [{formatted_values}]"


_FALLBACK_DECIMATE_TARGET = 9600


def _compute_decimate_target(screen) -> int:
    # return conservative fallback when no screen is available (headless / early startup)
    if screen is None:
        return _FALLBACK_DECIMATE_TARGET
    # physical pixels: width × clamped device-pixel ratio
    return screen.size().width() * max(5, int(screen.devicePixelRatio()))


class MainWindow(QMainWindow):

    def __init__(self, qraw_file: QRawFile, source_qraw_path: Path | None = None, start_empty: bool = False):
        super().__init__()
        # load and set the application icon
        icon = load_app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        # keep a reference to the source qraw object for creating secondary windows
        self._qraw_file = qraw_file
        # extract information from file
        self._default_chart_type = qraw_file.chart_type
        self._abscissa = qraw_file.abscissa
        self._abscissa_scale = qraw_file.abscissa_scale
        self._expression_manager = qraw_file.expression_manager
        self._step_information = qraw_file.step_information
        # normalize numpy scalar to built-in int for stable Qt property marshalling
        self._plot_suggestions = [] if start_empty else qraw_file.get_plot_suggestions()
        # store the simulation file path for use by the Jupyter integration
        self._qraw_path = source_qraw_path if source_qraw_path is not None else qraw_file.filename
        # set window title to include the loaded filename
        self.setWindowTitle(f"{self._default_chart_type} - {qraw_file.filename.name}")
        # apply dark background stylesheet to the window chrome
        self.setStyleSheet(f"QMainWindow {{ background: {_BG}; }}")
        # initialize data structures
        self._charts: list[Chart] = []
        # optional initial step selection applied when charts are first created (used by FFT windows to pre-focus on the same steps the user was viewing in the source chart)
        self._initial_selected_steps: set[int] | None = None
        # keep Jupyter windows alive to prevent garbage collection
        self._jupyter_windows: list[JupyterWindow] = []
        # default horizontal zoom
        self._abscissa_from_index = self._step_information.abscissa_from_index
        self._abscissa_to_index = self._step_information.abscissa_to_index
        # single QQuickView hosts the entire multi-chart scene — one Metal swap chain
        self._qml_view = QQuickView()
        self._qml_view.statusChanged.connect(self._on_qml_ready)
        self._qml_view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
        self._qml_view.setColor(QColor(_BG))
        self._qml_view.setSource(QUrl.fromLocalFile(str(_QML_FILE)))
        # embed the single QWindow into the main window's central area
        self._container = QWidget.createWindowContainer(self._qml_view, self)
        self.setCentralWidget(self._container)
        # create the native main menu structure
        self._create_main_menu()
        # decimation target — physical pixels of the primary screen width
        self._decimate_target = _compute_decimate_target(QGuiApplication.primaryScreen())
        # throttle timestamp for status bar updates
        self._last_status_time: float = 0.0

    def sizeHint(self):
        return QSize(1200, 800)

    def closeEvent(self, event) -> None:
        # release application-level child window ownership when this window closes
        _unregister_child_window(self)
        # delegate to the Qt base class when available
        close_event = getattr(super(), "closeEvent", None)
        if close_event is not None:
            close_event(event)

    @Slot(QQuickView.Status)
    def _on_qml_ready(self, status: QQuickView.Status):
        # only proceed once QML has finished loading successfully
        if status != QQuickView.Status.Ready:
            return
        # qml view root object
        self._root = self._qml_view.rootObject()
        # set window-level menu capability flags using built-in bool to avoid passing numpy.bool_ into QML properties
        self._root.setProperty("fftEnabled", bool(self._abscissa.unit == "s"))
        self._root.setProperty("stepToolEnabled", bool(self._step_information.length > 1))
        # connect signals from QML to Python handlers
        self._root.horizontalZoom.connect(self._on_horizontal_zoom)
        self._root.verticalZoom.connect(self._on_vertical_zoom)
        self._root.menuZoomToFit.connect(self._on_menu_zoom_to_fit)
        self._root.menuAutorange.connect(self._on_menu_autorange)
        self._root.menuZoomAbscissaExtent.connect(self._on_menu_zoom_abscissa_extent)
        self._root.menuAddRemovePlots.connect(self._on_menu_add_remove_plots)
        self._root.menuDeleteAllPlots.connect(self._on_menu_delete_all_plots)
        self._root.menuAddChart.connect(self._on_menu_add_chart)
        self._root.menuDeleteChart.connect(self._on_menu_delete_chart)
        self._root.menuNewWindow.connect(self._on_menu_new_window)
        self._root.menuFft.connect(self._on_menu_fft)
        self._root.menuStepTool.connect(self._on_menu_step_tool)
        # connect pointer hover signals to update the status bar
        self._root.pointerMoved.connect(self._on_pointer_moved)
        self._root.pointerExited.connect(self._on_pointer_exited)
        # populate charts after the event loop starts so the window is visible first
        QTimer.singleShot(0, self._populate_charts)
        # log screen information for debugging purposes
        if logger.isEnabledFor(logging.DEBUG):
            QTimer.singleShot(0, self._log_screen_info)

    def _create_main_menu(self):
        # menu bar
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        # File | Open
        open_action = QAction("Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_menu_open_file)
        file_menu.addAction(open_action)

        # File | Quit
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        # Tools | Open in JupyterLab
        jupyter_action = QAction("Open in JupyterLab...", self)
        jupyter_action.triggered.connect(self._on_open_jupyter)
        tools_menu.addAction(jupyter_action)

        # Window menu
        window_menu = menu_bar.addMenu("&Window")
        # Window | Add Chart
        add_chart_action = QAction("Add Chart", self)
        add_chart_action.triggered.connect(lambda: self._on_menu_add_chart(len(self._charts) - 1))
        window_menu.addAction(add_chart_action)
        # Window | New Window
        new_window_action = QAction("New Window", self)
        new_window_action.triggered.connect(self._on_menu_new_window)
        window_menu.addAction(new_window_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        # Help | About
        about_action = QAction("About", self)
        about_action.triggered.connect(lambda: None)
        help_menu.addAction(about_action)

    @Slot()
    def _on_open_jupyter(self) -> None:
        # create a new Jupyter window for the currently loaded simulation file
        window = JupyterWindow(self._qraw_path)
        # keep a reference to prevent the window from being garbage collected
        self._jupyter_windows.append(window)
        # show a status bar message while the server starts; clear it when the window appears
        window.ready.connect(lambda: self.statusBar().clearMessage())
        # window.show() is intentionally absent — JupyterWindow shows itself when fully loaded

    def _populate_charts(self):
        # fall back to one empty chart when there are none
        if not self._plot_suggestions:
            # add a single chart with the default type for this file, but no series (empty)
            self._add_chart(self._default_chart_type, [])
            # exit
            return
        # loop suggestions — each suggestion carries its own chart type
        for suggestion in self._plot_suggestions:
            # append chart using the type encoded in the suggestion
            self._add_chart(suggestion.chart_type, suggestion.expressions)

    def _log_screen_info(self):
        # screen reference
        screen = self.screen()
        # log information
        logger.debug("Screen information:")
        logger.debug("Screen name: %s", screen.name())
        logger.debug("Screen size: %d x %d", screen.size().width(), screen.size().height())
        logger.debug("Device pixel ratio: %f", screen.devicePixelRatio())
        logger.debug("Refresh rate: %f", screen.refreshRate())

    def _add_chart(self, chart_type: str, expressions: list[Expression]):
        # chart index
        chart_index = len(self._charts)
        # create chart ui component in QML
        self._root.addChart()
        # get a reference to the chart's QML object so we can manipulate it
        chart_root = self._root.getChart(chart_index)
        # create chart instance
        chart = Chart(chart_root, chart_type, self._expression_manager, self._abscissa, self._abscissa_from_index, self._abscissa_to_index, self._step_information, self._decimate_target)
        # apply initial step selection when provided (e.g. FFT window inheriting source chart visibility)
        if self._initial_selected_steps is not None:
            chart.selected_steps = self._initial_selected_steps
        # add it to the list of charts so we can keep track of it
        self._charts.append(chart)
        # render chart
        chart.render("", self._abscissa_scale.value, set(expressions))

    @Slot(int, float, float, float)
    def _on_horizontal_zoom(self, chart_index: int, x_left_ratio: float, x_right_ratio: float, zoom_factor: float):
        # calculate horizontal axis indices from the supplied ratios
        total = len(self._abscissa.data)
        from_index = max(0, min(int(self._abscissa_from_index + x_left_ratio * (self._abscissa_to_index - self._abscissa_from_index)), total - 1))
        to_index = max(0, min(int(self._abscissa_from_index + x_right_ratio * (self._abscissa_to_index - self._abscissa_from_index)), total))
        # allow zoom-in beyond pixel width, only enforce a minimum window of 2 points
        min_window = 2
        # detect pure pan (translation) gestures: when the ratio span equals 1.0
        ratio_span = x_right_ratio - x_left_ratio
        # current window before the operation
        current_from = self._abscissa_from_index
        current_to = self._abscissa_to_index
        current_window = current_to - current_from
        # small epsilon for floating comparisons
        if abs(ratio_span - 1.0) < 1e-9 or zoom_factor == 1.0:
            # this is a pan: compute integer shift in samples and apply
            shift = int(round(x_left_ratio * current_window))
            new_from = max(0, min(total - current_window, current_from + shift))
            new_to = new_from + current_window
            from_index = new_from
            to_index = new_to
        else:
            # choose direction based on factor (<1 zoom-in, >1 zoom-out)
            window = to_index - from_index
            mid = (from_index + to_index) // 2
            step = max(1, window // 8)
            if window < min_window:
                from_index = max(0, mid - min_window // 2)
                to_index = min(total, from_index + min_window)
            elif zoom_factor > 1.0:
                # zoom-out: expand window by a small step, up to full range
                new_window = min(total, window + step)
                from_index = max(0, mid - new_window // 2)
                to_index = min(total, from_index + new_window)
            else:
                # zoom-in: reduce window by a small step, down to min_window
                new_window = max(min_window, window - step)
                from_index = max(0, mid - new_window // 2)
                to_index = min(total, from_index + new_window)
        # update fields
        self._abscissa_from_index = from_index
        self._abscissa_to_index = to_index
        # update all charts — horizontal zoom is shared across all panels
        for chart in self._charts:
            # update zoom window — pass None for Y to leave per-chart vertical zoom unchanged
            chart.update_zoom_window(from_index, to_index, None, None)

    @Slot(int, float, float)
    def _on_vertical_zoom(self, chart_index: int, y_top_ratio: float, y_bottom_ratio: float):
        # find chart at index
        chart = self._charts[chart_index]
        # update vertical zoom window only — pass -1 for horizontal indices to leave them unchanged
        chart.update_zoom_window(-1, -1, y_top_ratio, y_bottom_ratio)

    @Slot(int)
    def _on_menu_zoom_to_fit(self, chart_index: int):
        # log information
        logger.debug("User requested zoom to fit on chart at index: %d", chart_index)
        # reset horizontal axis indices to show the full range of the abscissa
        self._abscissa_from_index = self._step_information.abscissa_from_index
        self._abscissa_to_index = self._step_information.abscissa_to_index
        # update charts
        for index, chart in enumerate(self._charts):
            # check if this is the chart that triggered the zoom to fit action
            if index == chart_index:
                # reset zoom window
                chart.reset_zoom_window(self._abscissa_from_index, self._abscissa_to_index, 0.0, 1.0)
                # next
                continue
            # update horizontal zoom window only, keep vertical zoom as is
            chart.update_zoom_window(self._abscissa_from_index, self._abscissa_to_index, None, None)

    @Slot(int)
    def _on_menu_autorange(self, chart_index: int):
        # log information
        logger.debug("User requested autorange on chart at index: %d", chart_index)
        # find chart at index
        chart = self._charts[chart_index]
        # reset zoom window
        chart.reset_zoom_window(-1, -1, 0.0, 1.0)

    @Slot(int)
    def _on_menu_zoom_abscissa_extent(self, chart_index: int):
        # log information
        logger.debug("User requested zoom abscissa extent on chart at index: %d", chart_index)
        # update fields
        self._abscissa_from_index = self._step_information.abscissa_from_index
        self._abscissa_to_index = self._step_information.abscissa_to_index
        # update charts
        for chart in self._charts:
            # update zoom window
            chart.reset_zoom_window(self._abscissa_from_index, self._abscissa_to_index, None, None)

    @Slot(int)
    def _on_menu_add_remove_plots(self, chart_index: int):
        # log information
        logger.debug("User requested adding/removing plots on chart at index: %d", chart_index)
        # find chart at index
        chart = self._charts[chart_index]
        # open the add plot dialog
        dialog = AddPlotDialog(self._expression_manager, chart.expressions, self)
        # exit if the user cancelled
        if dialog.exec() != AddPlotDialog.DialogCode.Accepted:
            return
        # plot selected expressions on the chart
        chart.plot_series(dialog.selected_expressions)
        # auto range axes to include the newly added series (wait for QT event loop)
        QTimer.singleShot(250, lambda: (chart.auto_range()))

    @Slot(int)
    def _on_menu_delete_all_plots(self, chart_index: int):
        # log information
        logger.debug("User requested deleting all plots on chart at index: %d", chart_index)
        # find chart
        chart = self._charts[chart_index]
        # clear chart
        chart.clear()

    @Slot(int)
    def _on_menu_add_chart(self, chart_index: int):
        # log information
        logger.debug("User requested adding a new chart at index: %d", chart_index)
        # add a new chart with no pre-populated expressions
        self._add_chart(self._default_chart_type, [])

    @Slot(int)
    def _on_menu_delete_chart(self, chart_index: int):
        # log information
        logger.debug("User requested deleting chart at index: %d", chart_index)
        # delete chart at index (do ot swap these two statements, C++ objects get deleted immediately when their Python reference is deleted, so we need to remove the chart from the UI before deleting the Python object)
        self._root.removeChart(chart_index)
        del self._charts[chart_index]

    @Slot()
    def _on_menu_new_window(self):
        # log information
        logger.debug("User requested opening a new window")
        # create a new independent main window sharing the same source data and path
        new_window = MainWindow(self._qraw_file, source_qraw_path=self._qraw_path, start_empty=True)
        # keep reference alive independently of the source main window
        _register_child_window(new_window)
        # show the new window
        new_window.show()

    @Slot()
    def _on_menu_open_file(self) -> None:
        # use application-level lock so only one open-file dialog can exist at a time
        global _OPEN_FILE_DIALOG_ACTIVE
        if _OPEN_FILE_DIALOG_ACTIVE:
            return
        _OPEN_FILE_DIALOG_ACTIVE = True
        # prompt for a file
        try:
            input_path, _ = QFileDialog.getOpenFileName(self, "Open QRAW File", "", "QRAW Files (*.qraw);;All Files (*)")
        finally:
            _OPEN_FILE_DIALOG_ACTIVE = False
        # exit when the user cancels the dialog
        if not input_path:
            return
        # load file and create a new main window through shared open path
        new_window = open_qraw_as_window(input_path, lambda qraw_file: MainWindow(qraw_file))
        # exit when the file fails to load
        if new_window is None:
            return
        # keep reference alive independently of the source main window
        _register_child_window(new_window)
        # show the new window
        new_window.show()

    @Slot(int)
    def _on_menu_step_tool(self, chart_index: int):
        # log information
        logger.debug("User requested step tool on chart at index: %d", chart_index)
        # check the chart index is valid
        if chart_index < 0 or chart_index >= len(self._charts):
            return
        # current chart
        chart = self._charts[chart_index]
        # get selected steps for this chart, make a copy
        selected_steps = set(chart.selected_steps)
        # open step tool dialog
        dialog = StepToolDialog(self, self._step_information, selected_steps)
        # exit if the user canceled
        if dialog.exec() != StepToolDialog.DialogCode.Accepted:
            return
        # store chart-local selected steps for later filtering phase
        chart.selected_steps = dialog.selected_steps
        # auto range axes
        chart.auto_range()

    @Slot(int)
    def _on_menu_fft(self, chart_index: int):
        # log information
        logger.debug("User requested FFT on chart at index: %d", chart_index)
        # find chart
        chart = self._charts[chart_index]
        # collect real-valued ordinate expressions currently plotted on this chart
        expressions = [v for v in chart.expressions if not v.complex and v != chart.abscissa]
        if not expressions:
            # log warning — no suitable expressions to transform
            logger.warning("No suitable time-domain expressions to FFT on chart %d", chart_index)
            # exit
            return
        # open FFT settings dialog
        dialog = FftDialog(expressions, self._abscissa, self._abscissa_from_index, self._abscissa_to_index, self)
        if dialog.exec() != FftDialog.DialogCode.Accepted:
            return
        # retrieve user selections
        result_expressions = dialog.result_expressions
        from_index = dialog.result_from_index
        to_index = dialog.result_to_index
        window = dialog.result_window
        zero_pad = dialog.result_zero_pad
        normalize = dialog.result_normalize
        keep_dc = dialog.result_keep_dc
        output = dialog.result_output
        # number of samples per step (abscissa is already trimmed to one step)
        step_points = len(self._abscissa.data)
        # shared abscissa slice — identical across all steps since the abscissa is periodic
        x = self._abscissa.data[from_index:to_index]
        # build one batch matrix for all expressions and all steps in expression-major order;
        # rows are [expr0-step0, expr0-step1, ..., exprN-stepS]
        signal_count = len(result_expressions)
        sample_count = to_index - from_index
        # allocate dense batch buffer once; this owns the copied FFT input payload
        y_matrix = np.empty((signal_count * self._steps, sample_count))
        # fill each expression block with all step slices at once
        for expr_index, expression in enumerate(result_expressions):
            row_start = expr_index * self._steps
            row_end = row_start + self._steps
            # reshape + slice are view operations when source layout is step-major contiguous
            step_matrix = expression.data.reshape(self._steps, step_points)[:, from_index:to_index]
            # copy step block into the preallocated batch buffer
            y_matrix[row_start:row_end] = step_matrix
        try:
            # fft internals allocate output arrays (spectrum/frequency/value matrices)
            frequencies, fft_matrix = compute_fft_many(x, y_matrix, window, zero_pad, normalize, output, keep_dc)
        except ValueError:
            # log exception and abort
            logger.exception("Batch FFT computation failed for chart %d", chart_index)
            # exit
            return
        # guard against an unexpectedly empty frequency axis
        if len(frequencies) == 0:
            # log error and abort when no frequencies are returned
            logger.error("FFT computation returned an empty frequency axis for chart %d", chart_index)
            # exit
            return
        # number of frequency bins produced per step
        freq_points = len(frequencies)
        # build FFT expressions from row slices in the batched output matrix;
        # each expression keeps step-major contiguous layout: [step0, step1, ..., stepS]
        fft_expressions: list[Expression] = []
        for expr_index, expression in enumerate(result_expressions):
            row_start = expr_index * self._steps
            row_end = row_start + self._steps
            # row slice is a view; reshape keeps a view for contiguous row blocks
            expression_data = fft_matrix[row_start:row_end].reshape(self._steps * freq_points)
            expression_unit = "°" if output == FftOutput.PHASE else ("dB" if output == FftOutput.MAGNITUDE_DB else expression.unit)
            fft_expressions.append(Expression(f"FFT({expression.name.replace(' ', '')})", expression_data, expression_unit))
        # create frequency expression — length matches a single step period (freq_points)
        freq_expression = Expression("Frequency", frequencies, "Hz")
        # build expression manager with frequency abscissa and all FFT results
        expression_manager = ExpressionManager([freq_expression] + fft_expressions)
        # build one «name» group per FFT expression so each gets its own chart
        plot_suggestion = " ".join(f"\xabfft {e.name}\xbb" for e in fft_expressions)
        # create a synthetic QRawFile with frequency abscissa and all FFT values;
        # steps matches the source simulation so the FFT window inherits full step coverage
        fft_qraw = QRawFile(filename=self._qraw_path, title=f"FFT \u2013 {', '.join(e.name for e in result_expressions)}", date="", plotname="FFT", complex=False, steps=self._steps, abscissa=freq_expression, abscissa_scale=AbscissaScale.LINEAR, command="", plot_suggestion=plot_suggestion, expression_manager=expression_manager, chart_type="FFT")
        # create a new MainWindow to render the FFT result using the existing infrastructure;
        # pass the original source path so Jupyter always opens the correct .qraw file
        fft_window = MainWindow(fft_qraw, source_qraw_path=self._qraw_path)
        # pre-focus the FFT window on the same steps the user was viewing in the source chart
        fft_window._initial_selected_steps = chart.selected_steps
        # keep reference alive independently of the source main window
        _register_child_window(fft_window)
        # show the FFT result window
        fft_window.show()

    @Slot(int, float)
    def _on_pointer_moved(self, chart_index: int, x_ratio: float):
        # throttle updates to ~30 fps to avoid saturating the UI thread
        now = time.monotonic()
        if now - self._last_status_time < _MIN_STATUS_INTERVAL:
            return
        # update timestamp of last status update
        self._last_status_time = now
        # guard against invalid chart index
        if chart_index < 0 or chart_index >= len(self._charts):
            return
        # chart at index
        chart = self._charts[chart_index]
        # compute abscissa index within the current zoom window
        idx = chart.sample_index_at_ratio(x_ratio)
        # retrieve the stored abscissa value (may be in log space for decade/octave scales)
        x_stored = float(self._abscissa.data[idx])
        # convert stored value back to physical abscissa value
        if self._abscissa_scale == AbscissaScale.DECADE:
            x_actual = 10 ** x_stored
        elif self._abscissa_scale == AbscissaScale.OCTAVE:
            x_actual = 2 ** x_stored
        else:
            x_actual = x_stored
        # append abscissa value
        parts = [_format_values(self._abscissa.name, [x_actual], self._abscissa.unit)]
        # process samples from chart
        for name, unit, values in chart.sample_at(x_ratio):
            parts.append(_format_values(name, values, unit))
        # update status bar with the composed string
        self.statusBar().showMessage("    ".join(parts))

    @Slot(int)
    def _on_pointer_exited(self, chart_index: int):
        self.statusBar().clearMessage()
