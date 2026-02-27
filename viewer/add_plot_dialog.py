import logging
from pathlib import Path

from PySide6.QtCore import QUrl, Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtQml import QJSValue
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

from .variable import Variable

logger = logging.getLogger(__name__)

_QML_FILE = Path(__file__).parent / "add_plot_dialog.qml"
_BG = "#1a1b1e"


class AddPlotDialog(QDialog):

    def __init__(self, variables: list[Variable], selected_variables: list[Variable], parent=None):
        super().__init__(parent)
        # store variables for index mapping when reading back the selection
        self._variables = variables
        self._selected_variables: set[Variable] = set(selected_variables)
        # window setup
        self.setWindowTitle("Add Plot")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(560, 420)
        # create the QML view — inject variable names before loading so QML can bind to them immediately
        self._qml_view = QQuickView()
        self._qml_view.statusChanged.connect(self._on_qml_ready)
        self._qml_view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
        self._qml_view.setColor(QColor(_BG))
        self._qml_view.rootContext().setContextProperty("variableNames", [v.name for v in variables])
        self._qml_view.setSource(QUrl.fromLocalFile(str(_QML_FILE)))
        # embed the QML view into the dialog
        container = QWidget.createWindowContainer(self._qml_view, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)

    @Slot(QQuickView.Status)
    def _on_qml_ready(self, status: QQuickView.Status):
        # only proceed once QML has finished loading successfully
        if status != QQuickView.Status.Ready:
            return
        # connect QML dialog signals to Python accept / reject
        root = self._qml_view.rootObject()
        # signals
        root.dialogAccepted.connect(self.accept)
        root.dialogRejected.connect(self.reject)
        root.selectionChanged.connect(self._on_selection_changed)
        # initialize view
        root.initialize([[v.name, v in self._selected_variables] for v in self._variables])

    @Slot(str, bool)
    def _on_selection_changed(self, expression: str, selected: bool):
        # log information
        logger.debug("User %s expression: %s", "selected" if selected else "deselected", expression)
        # handle selection change from QML — update the selected variables list
        variable = next((v for v in self._variables if v.name == expression), None)
        # toggle variable in selected list based on selection state
        if variable is not None:
            # if selected, add to selected variables if not already there; if deselected, remove from selected variables if present
            if selected:
                # append variable to selected set
                self._selected_variables.add(variable)
                # exit
                return
            # remove variable from selected set
            self._selected_variables.remove(variable)

    @property
    def selected_variables(self) -> set[Variable]:
        return self._selected_variables
