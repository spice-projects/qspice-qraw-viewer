import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QColor
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

from .expression import Expression
from .fft import FftOutput, WindowFunction, ZeroPadding, fft_frequency_range

logger = logging.getLogger(__name__)

_QML_FILE = Path(__file__).parent / "fft_dialog.qml"
_BG = "#1a1b1e"


class FftDialog(QDialog):
    """Dialog for configuring and launching an FFT computation.

    The dialog exposes a QML UI that lets the user choose a variable, data
    range, window function, zero-padding, normalisation and output format.
    After acceptance the result properties hold the user's selections.

    Parameters
    ----------
    variables        : ordinate expressions available for FFT (abscissa excluded).
    abscissa         : time-domain abscissa expression.
    zoom_from_index  : left edge of the current visible zoom window (sample index).
    zoom_to_index    : right edge of the current visible zoom window (sample index).
    parent           : optional parent widget.
    """

    def __init__(self, variables: list[Expression], abscissa: Expression, zoom_from_index: int, zoom_to_index: int, parent=None):
        super().__init__(parent)
        # store references
        self._variables = variables
        self._abscissa = abscissa
        self._zoom_from_index = zoom_from_index
        self._zoom_to_index = zoom_to_index
        # result fields populated when the dialog is accepted
        self._result_variable: Expression | None = None
        self._result_from_index: int = 0
        self._result_to_index: int = len(abscissa.values)
        self._result_window: WindowFunction = WindowFunction.RECTANGULAR
        self._result_zero_pad: ZeroPadding = ZeroPadding.NONE
        self._result_normalize: bool = False
        self._result_output: FftOutput = FftOutput.MAGNITUDE
        # window setup
        self.setWindowTitle("FFT")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(480, 540)
        # compute frequency-range preview from the full abscissa
        abscissa_values = abscissa.values
        df, f_nyquist = fft_frequency_range(abscissa_values)
        # expose data to QML via context properties
        # build the initial property values for the QML root object
        self._ctx_properties = {
            "variableNames": [v.name for v in variables],
            "windowFunctions": [w.value for w in WindowFunction],
            "outputTypes": [o.value for o in FftOutput],
            "zeroPaddingOptions": [z.value for z in ZeroPadding],
            "freqRangePreview": f"0 Hz – {f_nyquist:.4g} Hz (Nyquist)",
            "binWidthPreview": f"{df:.4g} Hz / bin",
            "abscissaMin": float(abscissa_values[0]),
            "abscissaMax": float(abscissa_values[-1]),
            "zoomFromTime": float(abscissa_values[min(zoom_from_index, len(abscissa_values) - 1)]),
            "zoomToTime": float(abscissa_values[min(zoom_to_index, len(abscissa_values)) - 1]),
        }
        # create QML view
        self._qml_view = QQuickView()
        self._qml_view.statusChanged.connect(self._on_qml_ready)
        self._qml_view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
        self._qml_view.setColor(QColor(_BG))
        self._qml_view.setSource(QUrl.fromLocalFile(str(_QML_FILE)))
        # embed QML view into dialog
        container = QWidget.createWindowContainer(self._qml_view, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)

    @Slot(QQuickView.Status)
    def _on_qml_ready(self, status: QQuickView.Status):
        # only proceed once QML has finished loading successfully
        if status != QQuickView.Status.Ready:
            return
        # set properties directly on the root object
        root = self._qml_view.rootObject()
        for key, value in self._ctx_properties.items():
            root.setProperty(key, value)
        # connect QML dialog signals to Python accept / reject
        root.dialogAccepted.connect(self._on_dialog_accepted)
        root.dialogRejected.connect(self.reject)

    @Slot(str, str, str, str, bool, str, float, float)
    def _on_dialog_accepted(self, variable_name: str, window_fn: str, zero_pad: str, output: str, normalize: bool, range_mode: str, custom_from: float, custom_to: float):
        # resolve variable by name against the stored list
        variable = next((v for v in self._variables if v.name == variable_name), None)
        if variable is None:
            # log warning and reject dialog when variable cannot be found
            logger.warning("FFT dialog accepted with unknown variable: %s", variable_name)
            self.reject()
            return
        # store the resolved variable
        self._result_variable = variable
        # window function
        try:
            self._result_window = WindowFunction(window_fn)
        except ValueError:
            # fall back to rectangular when the value is unrecognised
            self._result_window = WindowFunction.RECTANGULAR
        # zero-padding
        try:
            self._result_zero_pad = ZeroPadding(zero_pad)
        except ValueError:
            # fall back to no padding when the value is unrecognised
            self._result_zero_pad = ZeroPadding.NONE
        # output type
        try:
            self._result_output = FftOutput(output)
        except ValueError:
            # fall back to magnitude when the value is unrecognised
            self._result_output = FftOutput.MAGNITUDE
        # normalize flag
        self._result_normalize = normalize
        # data range
        abscissa_values = self._abscissa.values
        # total number of abscissa samples
        total = len(abscissa_values)
        if range_mode == "zoom":
            # use the current visible zoom window
            self._result_from_index = self._zoom_from_index
            self._result_to_index = self._zoom_to_index
        elif range_mode == "custom":
            # map user-supplied time values to nearest sample indices
            from_idx = int(np.searchsorted(abscissa_values, custom_from))
            to_idx = int(np.searchsorted(abscissa_values, custom_to, side="right"))
            # clamp to valid range ensuring at least 2 samples
            self._result_from_index = max(0, min(from_idx, total - 2))
            self._result_to_index = max(self._result_from_index + 2, min(to_idx, total))
        else:
            # use the full abscissa range
            self._result_from_index = 0
            self._result_to_index = total
        self.accept()

    @property
    def result_variable(self) -> Expression | None:
        return self._result_variable

    @property
    def result_from_index(self) -> int:
        return self._result_from_index

    @property
    def result_to_index(self) -> int:
        return self._result_to_index

    @property
    def result_window(self) -> WindowFunction:
        return self._result_window

    @property
    def result_zero_pad(self) -> ZeroPadding:
        return self._result_zero_pad

    @property
    def result_normalize(self) -> bool:
        return self._result_normalize

    @property
    def result_output(self) -> FftOutput:
        return self._result_output
