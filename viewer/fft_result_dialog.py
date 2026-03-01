import csv
import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGraphs import QLineSeries
from PySide6.QtGui import QColor
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QDialog, QFileDialog, QVBoxLayout, QWidget

from .fft import FftOutput

logger = logging.getLogger(__name__)

_QML_FILE = Path(__file__).parent / "fft_result_dialog.qml"
_BG = "#1a1b1e"


class FftResultDialog(QDialog):
    """Dialog that displays FFT results as a frequency-domain chart.

    Parameters
    ----------
    series_name  : name of the source variable (used for the title/label).
    output       : the FFT output type that determines the Y-axis label.
    frequencies  : 1-D numpy array of frequency values in Hz.
    values       : 1-D numpy array of spectrum values.
    parent       : optional parent widget.
    """

    def __init__(self, series_name: str, output: FftOutput, frequencies: np.ndarray, values: np.ndarray, parent=None):
        super().__init__(parent)
        # store fields
        self._series_name = series_name
        self._output = output
        self._frequencies = frequencies
        self._values = values
        # keep series alive so Qt GC does not collect it before rendering
        self._series: QLineSeries | None = None
        # dialog setup
        self.setWindowTitle(f"FFT – {series_name}")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(800, 500)
        # determine Y-axis unit label
        y_label = self._y_axis_label(output)
        # pass metadata to QML via context properties
        self._qml_view = QQuickView()
        self._qml_view.statusChanged.connect(self._on_qml_ready)
        self._qml_view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
        self._qml_view.setColor(QColor(_BG))
        # set individual context properties
        ctx = self._qml_view.rootContext()
        ctx.setContextProperty("seriesLabel", series_name)
        ctx.setContextProperty("yAxisLabel", y_label)
        ctx.setContextProperty("freqMin", float(frequencies[0]) if len(frequencies) > 0 else 0.0)
        ctx.setContextProperty("freqMax", float(frequencies[-1]) if len(frequencies) > 0 else 1.0)
        ctx.setContextProperty("yMin", float(np.min(values)) if len(values) > 0 else 0.0)
        ctx.setContextProperty("yMax", float(np.max(values)) if len(values) > 0 else 1.0)
        self._qml_view.setSource(QUrl.fromLocalFile(str(_QML_FILE)))
        # embed QML view
        container = QWidget.createWindowContainer(self._qml_view, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)

    @staticmethod
    def _y_axis_label(output: FftOutput) -> str:
        # dB magnitude uses the decibel unit
        if output == FftOutput.MAGNITUDE_DB:
            return "dB"
        # phase is expressed in degrees
        if output == FftOutput.PHASE:
            return "°"
        return ""

    @Slot(QQuickView.Status)
    def _on_qml_ready(self, status: QQuickView.Status):
        # only proceed once QML has finished loading successfully
        if status != QQuickView.Status.Ready:
            return
        # connect QML signals to Python handlers
        root = self._qml_view.rootObject()
        root.exportCsvRequested.connect(self._on_export_csv)
        root.closeRequested.connect(self.close)
        # create QLineSeries, fill it and hand it to the QML chart
        self._series = QLineSeries()
        # set line width
        self._series.setWidth(1)
        # set line color
        self._series.setColor(QColor("#f77f00"))
        # populate series with FFT data — ascontiguousarray ensures Qt-compatible memory layout
        self._series.replaceNp(np.ascontiguousarray(self._frequencies, dtype=np.float64), np.ascontiguousarray(self._values, dtype=np.float64))
        # add series to the QML chart
        root.addSeries(self._series)

    @Slot()
    def _on_export_csv(self):
        # open a native save dialog to let the user pick a destination file
        filename, _ = QFileDialog.getSaveFileName(self, "Export FFT to CSV", f"fft_{self._series_name}.csv", "CSV Files (*.csv);;All Files (*)")
        # exit early if the user cancelled without selecting a file
        if not filename:
            return
        try:
            # write frequency and spectrum value pairs as CSV rows
            with open(filename, "w", newline="") as f:
                # create writer
                writer = csv.writer(f)
                # header row with column labels
                writer.writerow(["Frequency (Hz)", self._output.value])
                # data rows — one row per frequency bin
                for freq, val in zip(self._frequencies, self._values):
                    writer.writerow([freq, val])
            # log success
            logger.info("FFT results exported to %s", filename)
        except OSError:
            # log error
            logger.exception("Failed to export FFT results to %s", filename)
