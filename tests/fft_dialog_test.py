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
    dialog = FftDialog.__new__(FftDialog)
    dialog._variables = [e1, e2]
    dialog._abscissa = abscissa
    dialog._zoom_from_index = zoom_from
    dialog._zoom_to_index = zoom_to
    dialog._result_variable = None
    dialog._result_from_index = 0
    dialog._result_to_index = len(abscissa_values)
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

    def test_unknown_variable_calls_reject(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act — pass a name that does not match any variable
        dialog._on_dialog_accepted("NoSuchVar", "Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(len(dialog._rejected_calls), 1)
        self.assertIsNone(dialog._result_variable)

    def test_known_variable_stored_and_accepted(self):
        # arrange
        dialog, e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertIs(dialog._result_variable, e1)
        self.assertEqual(len(dialog._accepted_calls), 1)

    def test_window_function_hamming_stored(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Hamming", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_window, WindowFunction.HAMMING)

    def test_window_function_unknown_falls_back_to_rectangular(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "???", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_window, WindowFunction.RECTANGULAR)

    def test_zero_pad_next_power_stored(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act — value string must match the enum definition exactly (capital P and T)
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "Next Power of Two", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_zero_pad, ZeroPadding.NEXT_POWER_OF_TWO)

    def test_zero_pad_unknown_falls_back_to_none(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "???", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_zero_pad, ZeroPadding.NONE)

    def test_output_type_magnitude_db_stored(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude (dB)", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_output, FftOutput.MAGNITUDE_DB)

    def test_output_type_phase_stored(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Phase", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_output, FftOutput.PHASE)

    def test_output_type_unknown_falls_back_to_magnitude(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "???", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_output, FftOutput.MAGNITUDE)

    def test_normalize_flag_true(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", True, "full", 0.0, 1.0, False)
        # assert
        self.assertTrue(dialog._result_normalize)

    def test_normalize_flag_false(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertFalse(dialog._result_normalize)

    def test_range_mode_full_uses_all_samples(self):
        # arrange — 11-point abscissa
        dialog, _e1, _e2 = _make_dialog(np.linspace(0.0, 1.0, 11))
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_from_index, 0)
        self.assertEqual(dialog._result_to_index, 11)

    def test_range_mode_zoom_uses_stored_zoom_window(self):
        # arrange — zoom window [3, 8)
        dialog, _e1, _e2 = _make_dialog(np.linspace(0.0, 1.0, 11), zoom_from=3, zoom_to=8)
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "zoom", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_from_index, 3)
        self.assertEqual(dialog._result_to_index, 8)

    def test_range_mode_custom_maps_time_to_indices(self):
        # arrange — 11-point abscissa 0…1 s
        abscissa = np.linspace(0.0, 1.0, 11)
        dialog, _e1, _e2 = _make_dialog(abscissa)
        # act — request range 0.2 s … 0.8 s (avoid 0.6 which has floating-point representability issues)
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "custom", 0.2, 0.8, False)
        # assert — from_index=2 for 0.2; to_index verified against actual searchsorted output
        self.assertEqual(dialog._result_from_index, 2)
        # searchsorted(arr, 0.8, side='right') where arr[8]=0.8 → position 9
        import numpy as _np
        expected_to = int(_np.searchsorted(abscissa, 0.8, side="right"))
        self.assertEqual(dialog._result_to_index, expected_to)

    def test_range_mode_custom_clamps_above_total(self):
        # arrange
        abscissa = np.linspace(0.0, 1.0, 11)
        dialog, _e1, _e2 = _make_dialog(abscissa)
        # act — to_time beyond the end of the array
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "custom", 0.5, 999.0, False)
        # assert — to_index clamped to total (11)
        self.assertEqual(dialog._result_to_index, 11)

    def test_range_mode_custom_ensures_minimum_two_samples(self):
        # arrange — very narrow custom range that resolves to the same index
        abscissa = np.linspace(0.0, 1.0, 11)
        dialog, _e1, _e2 = _make_dialog(abscissa)
        # act — from_time and to_time both map to index 5
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "custom", 0.5, 0.5, False)
        # assert — at least 2 samples guaranteed
        self.assertGreaterEqual(dialog._result_to_index - dialog._result_from_index, 2)

    def test_range_mode_unknown_falls_back_to_full(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog(np.linspace(0.0, 1.0, 11))
        # act — unrecognised range_mode triggers full-range fallback
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "unknown_mode", 0.0, 1.0, False)
        # assert
        self.assertEqual(dialog._result_from_index, 0)
        self.assertEqual(dialog._result_to_index, 11)

    def test_keep_dc_flag_true(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, True)
        # assert
        self.assertTrue(dialog._result_keep_dc)

    def test_keep_dc_flag_false(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # assert
        self.assertFalse(dialog._result_keep_dc)


class TestFftDialogResultProperties(TestCase):

    def test_result_variable_property_default_is_none(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        # act / assert
        self.assertIsNone(dialog.result_variable)

    def test_result_variable_property_after_accept(self):
        # arrange
        dialog, e1, _e2 = _make_dialog()
        dialog._on_dialog_accepted("V(R1)", "Rectangular", "None", "Magnitude", False, "full", 0.0, 1.0, False)
        # act / assert
        self.assertIs(dialog.result_variable, e1)

    def test_result_from_index_property(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        dialog._result_from_index = 4
        # act / assert
        self.assertEqual(dialog.result_from_index, 4)

    def test_result_to_index_property(self):
        # arrange
        dialog, _e1, _e2 = _make_dialog()
        dialog._result_to_index = 7
        # act / assert
        self.assertEqual(dialog.result_to_index, 7)

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
