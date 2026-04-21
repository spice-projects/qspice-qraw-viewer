import sys
import importlib
from unittest import TestCase
from unittest.mock import MagicMock

# mock PySide6 submodules before importing step_tool_dialog, which requires Qt at import time
sys.modules["PySide6"] = MagicMock()
sys.modules["PySide6.QtCore"] = MagicMock()
sys.modules["PySide6.QtGui"] = MagicMock()
sys.modules["PySide6.QtQuick"] = MagicMock()
sys.modules["PySide6.QtWidgets"] = MagicMock()
# Slot must act as a pass-through decorator so @Slot(...) does not replace the method with a mock
sys.modules["PySide6.QtCore"].Slot = lambda *a, **kw: (lambda f: f)
# QDialog must be a concrete class so that StepToolDialog can genuinely inherit from it
sys.modules["PySide6.QtWidgets"].QDialog = type(
    "QDialog",
    (),
    {
        "__init__": lambda self, parent=None: None,
        "accept": lambda self: None,
        "reject": lambda self: None,
        "setWindowTitle": lambda self, title: None,
        "setWindowModality": lambda self, modality: None,
        "resize": lambda self, width, height: None,
        "setMinimumHeight": lambda self, height: None,
        "setMinimumWidth": lambda self, width: None,
    },
)

from viewer.qraw_file import StepInformation  # noqa: E402
sys.modules.pop("viewer.step_tool_dialog", None)
StepToolDialog = importlib.import_module("viewer.step_tool_dialog").StepToolDialog  # noqa: E402


class TestStepToolDialog(TestCase):

    def test_constructor_builds_context_properties_from_step_information(self):
        # arrange
        class _Signal:
            def connect(self, _slot):
                return None
        class _FakeQuickView:
            class Status:
                Ready = object()
            class ResizeMode:
                SizeRootObjectToView = object()
            def __init__(self):
                self.statusChanged = _Signal()
            def setResizeMode(self, _mode):
                return None
            def setColor(self, _color):
                return None
            def setSource(self, _source):
                return None
            def rootObject(self):
                return MagicMock()
        class _FakeWidget:
            @staticmethod
            def createWindowContainer(_view, _parent):
                return object()
        class _FakeLayout:
            def __init__(self, _parent):
                return None
            def setContentsMargins(self, _a, _b, _c, _d):
                return None
            def addWidget(self, _widget):
                return None
        original_qquickview = StepToolDialog.__init__.__globals__["QQuickView"]
        original_qwidget = StepToolDialog.__init__.__globals__["QWidget"]
        original_qvboxlayout = StepToolDialog.__init__.__globals__["QVBoxLayout"]
        original_qurl = StepToolDialog.__init__.__globals__["QUrl"]
        original_qcolor = StepToolDialog.__init__.__globals__["QColor"]
        original_qt = StepToolDialog.__init__.__globals__["Qt"]
        StepToolDialog.__init__.__globals__["QQuickView"] = _FakeQuickView
        StepToolDialog.__init__.__globals__["QWidget"] = _FakeWidget
        StepToolDialog.__init__.__globals__["QVBoxLayout"] = _FakeLayout
        StepToolDialog.__init__.__globals__["QUrl"] = MagicMock(fromLocalFile=lambda value: value)
        StepToolDialog.__init__.__globals__["QColor"] = lambda value: value
        StepToolDialog.__init__.__globals__["Qt"] = MagicMock(WindowModality=MagicMock(WindowModal=object()))
        step_information = StepInformation(keys=["temp", "vdd"], values=[(25, 1.8), (100, 3.3)], indices=[slice(0, 2), slice(2, 4)])
        # act
        dialog = StepToolDialog(parent=MagicMock(), step_information=step_information, selected_steps={3, 1})
        # assert
        self.assertEqual(dialog._ctx_properties["parameterNames"], ["temp", "vdd"])
        self.assertEqual(dialog._ctx_properties["stepRows"], [{"stepIndex": 0, "values": ["25", "1.8"]}, {"stepIndex": 1, "values": ["100", "3.3"]}])
        self.assertEqual(dialog._ctx_properties["initialSelectedSteps"], [1, 3])
        StepToolDialog.__init__.__globals__["QQuickView"] = original_qquickview
        StepToolDialog.__init__.__globals__["QWidget"] = original_qwidget
        StepToolDialog.__init__.__globals__["QVBoxLayout"] = original_qvboxlayout
        StepToolDialog.__init__.__globals__["QUrl"] = original_qurl
        StepToolDialog.__init__.__globals__["QColor"] = original_qcolor
        StepToolDialog.__init__.__globals__["Qt"] = original_qt

    def test_on_selection_changed_updates_selected_steps_set(self):
        # arrange
        dialog = StepToolDialog.__new__(StepToolDialog)
        dialog._selected_steps = {0}
        # act
        dialog._on_selection_changed(2, True)
        dialog._on_selection_changed(0, False)
        # assert
        self.assertEqual(dialog._selected_steps, {2})

    def test_on_dialog_accepted_calls_accept(self):
        # arrange
        dialog = StepToolDialog.__new__(StepToolDialog)
        accepted_calls = []
        dialog.accept = lambda: accepted_calls.append(True)
        # act
        dialog._on_dialog_accepted([0, 1])
        # assert
        self.assertEqual(len(accepted_calls), 1)

    def test_selected_steps_property_returns_internal_selected_steps(self):
        # arrange
        dialog = StepToolDialog.__new__(StepToolDialog)
        dialog._selected_steps = {1, 3}
        # act
        result = dialog.selected_steps
        # assert
        self.assertEqual(result, {1, 3})
