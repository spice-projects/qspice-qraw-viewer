import sys
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

# mock PySide6 submodules before importing fft_dialog, which requires Qt at import time
sys.modules.setdefault("PySide6", MagicMock())
sys.modules.setdefault("PySide6.QtCore", MagicMock())
sys.modules.setdefault("PySide6.QtGui", MagicMock())
sys.modules.setdefault("PySide6.QtQuick", MagicMock())
sys.modules.setdefault("PySide6.QtWidgets", MagicMock())
# Slot must act as a pass-through decorator so @Slot(...) does not replace the method with a mock
sys.modules["PySide6.QtCore"].Slot = lambda *a, **kw: (lambda f: f)
# QDialog must be a concrete class so that FftDialog can genuinely inherit from it
sys.modules["PySide6.QtWidgets"].QDialog = type("QDialog", (), {"accept": lambda self: None, "reject": lambda self: None})

from viewer.expression import Expression  # noqa: E402
from viewer.fft import FftOutput, WindowFunction, ZeroPadding  # noqa: E402
from viewer.fft_dialog import FftDialog  # noqa: E402


def _make_dialog(abscissa_values=None, zoom_from=0, zoom_to=10):
    # build a FftDialog bypassing __init__ so no Qt objects are created
    if abscissa_values is None:
        abscissa_values = np.linspace(0.0, 1.0, 11)
    abscissa = Expression("Time", abscissa_values, "s")
    e1 = Expression("V(R1)", np.ones(len(abscissa_values)), "V")
    e2 = Expression("I(L1)", np.ones(len(abscissa_values)) * 2.0, "A")
    dialog = object.__new__(FftDialog)
    dialog._expressions = [e1, e2]
    dialog._abscissa = abscissa
    dialog._zoom_from_index = zoom_from
    dialog._zoom_to_index = zoom_to
    dialog._selected_expressions = {e1, e2}
    dialog._result_expressions = []
    dialog._result_from_index = float(abscissa_values[0])
    dialog._result_to_index = float(abscissa_values[-1])
    dialog._result_window = WindowFunction.RECTANGULAR
    dialog._result_zero_pad = ZeroPadding.NONE
    dialog._result_normalize = False
    dialog._result_keep_dc = False
    dialog._result_output = FftOutput.MAGNITUDE
    dialog._accepted_calls = []
    dialog._rejected_calls = []
    dialog.accept = lambda: dialog._accepted_calls.append(True)
    dialog.reject = lambda: dialog._rejected_calls.append(True)
    return dialog, e1, e2


class TestFftDialogOnDialogAccepted(TestCase):

    def test_no_selection_calls_reject(self):
        # arrange — clear all pre-selected expressions so the dialog has nothing to compute
        dialog, _e1, _e2 = _make_dialog()
        dialog._selected_expressions = set()
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(len(dialog._rejected_calls), 1)
        self.assertEqual(dialog._result_expressions, [])

    def test_on_expression_selection_changed_adds_expression(self):
        # arrange
        dialog, e1, _e2 = _make_dialog()
        dialog._selected_expressions = set()
        # act
        dialog._on_expression_selection_changed("V(R1)", True)
        # assert
        self.assertIn(e1, dialog._selected_expressions)

    def test_on_expression_selection_changed_removes_expression(self):
        # arrange
        dialog, e1, _e2 = _make_dialog()
        # act
        dialog._on_expression_selection_changed("V(R1)", False)
        # assert
        self.assertNotIn(e1, dialog._selected_expressions)

    def test_on_expression_selection_changed_unknown_name_no_error(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act / assert — unknown name must not raise
        dialog._on_expression_selection_changed("NonExistentVar", True)

    def test_known_variable_stored_and_accepted(self):
        # arrange — both expressions are pre-selected by default
        dialog, e1, e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertIn(e1, dialog._result_expressions)
        self.assertIn(e2, dialog._result_expressions)
        self.assertEqual(len(dialog._accepted_calls), 1)

    def test_window_function_hamming_stored(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Hamming", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_window, WindowFunction.HAMMING)

    def test_window_function_unknown_falls_back_to_rectangular(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("???", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_window, WindowFunction.RECTANGULAR)

    def test_zero_pad_next_power_stored(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act — value string must match the enum definition exactly (capital P and T)
        dialog._on_dialog_accepted("Rectangular", "Next Power of Two", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_zero_pad, ZeroPadding.NEXT_POWER_OF_TWO)

    def test_zero_pad_unknown_falls_back_to_none(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Rectangular", "???", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_zero_pad, ZeroPadding.NONE)

    def test_output_type_magnitude_db_stored(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude (dB)", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_output, FftOutput.MAGNITUDE_DB)

    def test_output_type_phase_stored(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Phase", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_output, FftOutput.PHASE)

    def test_output_type_unknown_falls_back_to_magnitude(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "???", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_output, FftOutput.MAGNITUDE)

    def test_normalize_flag_true(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", True, "full", 0.0, 1.0, False)
        # assert
        self.assertTrue(dialog._result_normalize)

    def test_normalize_flag_false(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertFalse(dialog._result_normalize)

    def test_range_mode_full_uses_all_samples(self):
        # arrange — 11-point abscissa
        dialog, _e1, _e2 = _make_dialog(np.linspace(0.0, 1.0, 11))
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_from_index, 0.0)
        self.assertEqual(dialog._result_to_index, 1.0)

    def test_range_mode_zoom_uses_stored_zoom_window(self):
        # arrange — zoom window [0.3, 0.8]
        dialog, _e1, _e2 = _make_dialog(np.linspace(0.0, 1.0, 11), zoom_from=0.3, zoom_to=0.8)
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "zoom", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_from_index, 0.3)
        self.assertEqual(dialog._result_to_index, 0.8)

    def test_range_mode_custom_uses_clamped_values(self):
        # arrange — 11-point abscissa 0…1 s
        abscissa = np.linspace(0.0, 1.0, 11)
        dialog, _e1, _e2 = _make_dialog(abscissa)
        # act — request range 0.2 s … 0.8 s (avoid 0.6 which has floating-point representability issues)
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "custom", 0.2, 0.8, False)
        # assert — custom values are stored directly after domain clamping
        self.assertEqual(dialog._result_from_index, 0.2)
        self.assertEqual(dialog._result_to_index, 0.8)

    def test_range_mode_custom_clamps_above_total(self):
        # arrange
        abscissa = np.linspace(0.0, 1.0, 11)
        dialog, _e1, _e2 = _make_dialog(abscissa)
        # act — to_time beyond the end of the array
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "custom", 0.5, 999.0, False)
        # assert — upper bound clamped to the last abscissa value
        self.assertEqual(dialog._result_to_index, 1.0)

    def test_range_mode_custom_preserves_single_value_window(self):
        # arrange — exact single-value custom range
        abscissa = np.linspace(0.0, 1.0, 11)
        dialog, _e1, _e2 = _make_dialog(abscissa)
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "custom", 0.5, 0.5, False)
        # assert
        self.assertEqual(dialog._result_from_index, 0.5)
        self.assertEqual(dialog._result_to_index, 0.5)

    def test_range_mode_unknown_falls_back_to_full(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog(np.linspace(0.0, 1.0, 11))
        # act — unrecognised range_mode triggers full-range fallback
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "unknown_mode", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_from_index, 0.0)
        self.assertEqual(dialog._result_to_index, 1.0)

    def test_keep_dc_flag_true(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, True)
        # assert
        self.assertTrue(dialog._result_keep_dc)

    def test_keep_dc_flag_false(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertFalse(dialog._result_keep_dc)


class TestFftDialogResultProperties(TestCase):

    def test_result_expressions_property_default_is_empty_list(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act / assert
        self.assertEqual(dialog.result_expressions, [])

    def test_result_expressions_property_after_accept(self):
        # arrange — both expressions are pre-selected; verify both appear in result
        dialog, e1, e2 = _make_dialog()
        dialog._on_dialog_accepted("Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # act / assert
        self.assertEqual(dialog.result_expressions, [e1, e2])

    def test_result_from_index_property(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        dialog._result_from_index = 0.4
        # act / assert
        self.assertEqual(dialog.result_from_index, 0.4)

    def test_result_to_index_property(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        dialog._result_to_index = 0.7
        # act / assert
        self.assertEqual(dialog.result_to_index, 0.7)

    def test_result_window_property(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        dialog._result_window = WindowFunction.BLACKMAN
        # act / assert
        self.assertEqual(dialog.result_window, WindowFunction.BLACKMAN)

    def test_result_zero_pad_property(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        dialog._result_zero_pad = ZeroPadding.NEXT_POWER_OF_TWO
        # act / assert
        self.assertEqual(dialog.result_zero_pad, ZeroPadding.NEXT_POWER_OF_TWO)

    def test_result_normalize_property(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        dialog._result_normalize = True
        # act / assert
        self.assertTrue(dialog.result_normalize)

    def test_result_keep_dc_property_default_is_false(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act / assert
        self.assertFalse(dialog.result_keep_dc)

    def test_result_keep_dc_property_after_set(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        dialog._result_keep_dc = True
        # act / assert
        self.assertTrue(dialog.result_keep_dc)

    def test_result_output_property(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        dialog._result_output = FftOutput.PHASE
        # act / assert
        self.assertEqual(dialog.result_output, FftOutput.PHASE)
