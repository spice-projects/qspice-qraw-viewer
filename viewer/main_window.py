import logging
from pathlib import Path

from PySide6.QtCore import QSize, QTimer, QUrl, Slot
from PySide6.QtGui import QColor, QGuiApplication, QAction, QKeySequence
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QMainWindow, QWidget, QFileDialog

from .add_plot_dialog import AddPlotDialog
from .chart import Chart
from .qraw_file import QRawFile
from .variable import Variable

logger = logging.getLogger(__name__)

_QML_FILE = Path(__file__).parent / "main_window.qml"

# background color matching the chart dark theme
_BG = "#1a1b1e"


class MainWindow(QMainWindow):

    def __init__(self, qraw_file: QRawFile):
        super().__init__()
        # store file data for later use in the UI
        self.qraw_file = qraw_file
        # set window title to include the loaded filename
        self.setWindowTitle(f"QSPICE v2 - {qraw_file.filename.name}")
        # apply dark background stylesheet to the window chrome
        self.setStyleSheet(f"QMainWindow {{ background: {_BG}; }}")
        # initialize data structures
        self._charts: list[Chart] = []
        # default horizontal zoom
        self._abscissa_from_index = 0
        self._abscissa_to_index = self.qraw_file.abscissa_points
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
        screen = QGuiApplication.primaryScreen()
        self._decimate_target = screen.size().width() * max(5, int(screen.devicePixelRatio()))

    def sizeHint(self):
        return QSize(1200, 800)

    @Slot(QQuickView.Status)
    def _on_qml_ready(self, status: QQuickView.Status):
        # only proceed once QML has finished loading successfully
        if status != QQuickView.Status.Ready:
            return
        # qml view root object
        self._root = self._qml_view.rootObject()
        # connect signals from QML to Python handlers
        self._root.horizontalZoom.connect(self._on_horizontal_zoom)
        self._root.verticalZoom.connect(self._on_vertical_zoom)
        self._root.menuZoomToFit.connect(self._on_menu_zoom_to_fit)
        self._root.menuAutorange.connect(self._on_menu_autorange)
        self._root.menuZoomAbscissaExtent.connect(self._on_menu_zoom_abscissa_extent)
        self._root.menuAddRemovePlots.connect(self._on_menu_add_remove_plots)
        self._root.menuDeleteAllPlots.connect(self._on_menu_delete_all_plots)
        self._root.menuAddWindow.connect(self._on_menu_add_window)
        self._root.menuDeleteWindow.connect(self._on_menu_delete_window)
        # populate charts after the event loop starts so the window is visible first
        QTimer.singleShot(0, self._populate_charts)

    def _create_main_menu(self):
        # menu bar
        menuBar = self.menuBar()

        # File menu
        file_menu = menuBar.addMenu("&File")
        # File | Open
        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_file_open)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        # File | Quit
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Window menu
        window_menu = menuBar.addMenu("&Window")
        # Window | Add Window
        add_window_action = QAction("Add Window", self)
        add_window_action.triggered.connect(lambda: self._on_menu_add_window(len(self._charts)-1))
        window_menu.addAction(add_window_action)

        # Help menu
        help_menu = menuBar.addMenu("&Help")
        # Help | About
        about_action = QAction("About", self)
        about_action.triggered.connect(lambda: None)
        help_menu.addAction(about_action)

    def _on_file_open(self):
        # native file dialog to pick a QRAW file
        filename, _ = QFileDialog.getOpenFileName(self, "Open QRAW File", "", "QRAW Files (*.qraw);;All Files (*)")
        if not filename:
            return
        # parse qraw file
        new_file = QRawFile.load(filename)
        if new_file is None:
            # log information
            logger.error("Failed to load selected QRAW file: %s", filename)
            # exit
            return
        
    def _populate_charts(self):
        # this is a calculated field, do it once per file and cache it
        plot_suggestions = self.qraw_file.plot_suggestions
        # fall back to one empty chart when there are none
        if not plot_suggestions:
            # add a single chart with the default type for this file, but no series (empty)
            self._add_chart([])
            # exit
            return
        # loop suggestions — each suggestion carries its own chart type
        for suggestion in plot_suggestions:
            # append chart using the type encoded in the suggestion
            self._add_chart(suggestion.variables)

    def _add_chart(self, variables: list[Variable]):
        # chart index
        chart_index = len(self._charts)
        # abscissa
        abscissa = self.qraw_file.variables[0]
        # create chart ui component in QML
        self._root.addChart()
        # get a reference to the chart's QML object so we can manipulate it
        chart_root = self._root.getChart(chart_index)
        # create chart instance
        chart = Chart(chart_root, abscissa, self._abscissa_from_index, self._abscissa_to_index, self._decimate_target)
        # add it to the list of charts so we can keep track of it
        self._charts.append(chart)
        # render chart
        chart.render("", self.qraw_file.abscissa_scale.value, set(variables))

    @Slot(int, float, float, float)
    def _on_horizontal_zoom(self, chart_index: int, x_left_ratio: float, x_right_ratio: float, zoom_factor: float):
        # calculate horizontal axis indices from the supplied ratios
        # convert ratios to indices and clamp to valid bounds before using
        total = len(self.qraw_file.variables[0].values)
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
            # update zoom window — pass -1 for Y to leave per-chart vertical zoom unchanged
            chart.update_zoom_window(from_index, to_index, -1, -1)

    @Slot(int, float, float)
    def _on_vertical_zoom(self, chart_index: int, y_top_ratio: float, y_bottom_ratio: float):
        # log information
        logger.debug("User requested vertical zoom on chart at index: %d, y=[%.3f, %.3f]", chart_index, y_top_ratio, y_bottom_ratio)
        # find chart at index
        chart = self._charts[chart_index]
        # update vertical zoom window only — pass -1 for horizontal indices to leave them unchanged
        chart.update_zoom_window(-1, -1, y_top_ratio, y_bottom_ratio)

    @Slot(int)
    def _on_menu_zoom_to_fit(self, chart_index: int):
        # log information
        logger.debug("User requested zoom to fit on chart at index: %d", chart_index)
        # reset horizontal axis indices to show the full range of the abscissa
        self._abscissa_from_index = 0
        self._abscissa_to_index = len(self.qraw_file.variables[0].values)
        # update charts
        for index, chart in enumerate(self._charts):
            # check if this is the chart that triggered the zoom to fit action
            if index == chart_index:
                # reset zoom window
                chart.reset_zoom_window(self._abscissa_from_index, self._abscissa_to_index, 0.0, 1.0)
                # next
                continue
            # update horizontal zoom window only, keep vertical zoom as is (use -1 to indicate no change)
            chart.update_zoom_window(self._abscissa_from_index, self._abscissa_to_index, -1, -1)

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
        # abscissa
        abscissa = self.qraw_file.variables[0]
        # update fields
        self._abscissa_from_index = 0
        self._abscissa_to_index = len(abscissa.values)
        # update charts
        for chart in self._charts:
            # update zoom window
            chart.reset_zoom_window(self._abscissa_from_index, self._abscissa_to_index, -1, -1)

    @Slot(int)
    def _on_menu_add_remove_plots(self, chart_index: int):
        # log information
        logger.debug("User requested adding/removing plots on chart at index: %d", chart_index)
        # find chart at index
        chart = self._charts[chart_index]
        # open the add plot dialog — skip index 0 (abscissa)
        dialog = AddPlotDialog(self.qraw_file.variables[1:], chart.variables, self)
        # exit if the user cancelled
        if dialog.exec() != AddPlotDialog.DialogCode.Accepted:
            return
        # plot selected variables on the chart
        chart.plot_series(dialog.selected_variables)
        # auto range axes to include the newly added series (wait for QT event loop)
        QTimer.singleShot(100, lambda: (chart.auto_range()))

    @Slot(int)
    def _on_menu_delete_all_plots(self, chart_index: int):
        # log information
        logger.debug("User requested deleting all plots on chart at index: %d", chart_index)
        # find chart
        chart = self._charts[chart_index]
        # clear chart
        chart.clear()

    @Slot(int)
    def _on_menu_add_window(self, chart_index: int):
        # log information
        logger.debug("User requested adding a new window at index: %d", chart_index)
        # add a new chart with no pre-populated variables
        self._add_chart([])

    @Slot(int)
    def _on_menu_delete_window(self, chart_index: int):
        # log information
        logger.debug("User requested deleting chart at index: %d", chart_index)
        # delete chart at index (do ot swap these two statements, C++ objects get deleted immediately when their Python reference is deleted, so we need to remove the chart from the UI before deleting the Python object)
        self._root.removeChart(chart_index)
        del self._charts[chart_index]
