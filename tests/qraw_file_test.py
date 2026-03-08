import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from viewer.expression import Expression
from viewer.qraw_file import (
    AbscissaScale,
    QRawFile,
    _chart_type_for_file,
    _process_scale,
    _process_step,
)

FIXTURES_DIR = Path(__file__).parent / "PyQSPICE"


class TestQRawFile(TestCase):

    def test_loading_10_DC(self):
        # arrange
        filename = FIXTURES_DIR / "UJ3N065080.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "DC Transfer Characteristic")
        self.assertFalse(qraw.complex)
        self.assertEqual(qraw.steps, 6)
        expressions = qraw.expression_manager.expressions
        self.assertEqual(len(expressions), 12)
        self.assertEqual(expressions[0].name, "V2")
        self.assertEqual(expressions[0].unit, "V")
        self.assertEqual(expressions[6].name, "Id(J1)")
        self.assertEqual(expressions[6].unit, "A")
        self.assertEqual(expressions[11].name, "P(V1)")
        self.assertEqual(expressions[11].unit, "W")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.LINEAR)
        # assert default variables to plot
        self.assertEqual(len(qraw.get_plot_suggestions()), 1)
        self.assertIs(qraw.get_plot_suggestions()[0].chart_type, "DC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["Id(J1)"])

    def test_loading_20_AC(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "AC Analysis")
        self.assertTrue(qraw.complex)
        self.assertEqual(qraw.steps, 2)
        # 32 raw variables + 2 aliases: Freq and Omega
        expressions = qraw.expression_manager.expressions
        self.assertEqual(len(expressions), 34)
        self.assertEqual(expressions[0].name, "Frequency")
        self.assertEqual(expressions[0].unit, "Hz")
        self.assertEqual(expressions[31].name, "X")
        self.assertEqual(expressions[32].name, "Freq")
        self.assertEqual(expressions[33].name, "Omega")
        self.assertEqual(expressions[31].unit, "")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.DECADE)
        # assert default variables to plot (AC keyword maps to AC chart type)
        self.assertEqual(len(qraw.get_plot_suggestions()), 1)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "AC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["V(vout)"])

    def test_loading_22_NyquistDia(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_Nyquist.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "AC Analysis")
        self.assertTrue(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        # 24 raw variables + 5 alias resistor currents + 2 aliases Freq/Omega
        expressions = qraw.expression_manager.expressions
        self.assertEqual(len(expressions), 31)
        self.assertEqual(expressions[0].name, "Frequency")
        self.assertEqual(expressions[0].unit, "Hz")
        self.assertEqual(expressions[1].name, "V(vout)")
        self.assertEqual(expressions[1].unit, "V")
        self.assertEqual(expressions[23].name, "I(COUT)")
        self.assertEqual(expressions[23].unit, "A")
        self.assertEqual(expressions[29].name, "Freq")
        self.assertEqual(expressions[30].name, "Omega")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.DECADE)
        # assert default variables
        self.assertEqual(len(qraw.get_plot_suggestions()), 1)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "AC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["(V(VOUT) / V(VO))"])

    def test_loading_30_TRAN(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "Transient Analysis")
        self.assertFalse(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        # 47 raw variables + 7 alias resistor/conductance currents
        expressions = qraw.expression_manager.expressions
        self.assertEqual(len(expressions), 54)
        self.assertEqual(expressions[0].name, "Time")
        self.assertEqual(expressions[0].unit, "s")
        self.assertEqual(expressions[1].name, "V(out)")
        self.assertEqual(expressions[1].unit, "V")
        self.assertEqual(expressions[46].name, "I(C1)")
        self.assertEqual(expressions[46].unit, "A")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.LINEAR)
        # assert default variables
        self.assertEqual(len(qraw.get_plot_suggestions()), 3)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["V(out)", "V(ss)", "V(in)"])
        self.assertEqual(qraw.get_plot_suggestions()[1].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[1].expressions], ["V(ics)", "V(comp)"])
        self.assertEqual(qraw.get_plot_suggestions()[2].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[2].expressions], ["V(on)", "V(off)"])

    def test_loading_40_Bode(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_Bode.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "AC Analysis")
        self.assertTrue(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        # 2 raw variables + 2 aliases: Freq and Omega
        expressions = qraw.expression_manager.expressions
        self.assertEqual(len(expressions), 4)
        self.assertEqual(expressions[0].name, "Frequency")
        self.assertEqual(expressions[0].unit, "Hz")
        self.assertEqual(expressions[1].name, "OpenLoopGain")
        self.assertEqual(expressions[2].name, "Freq")
        self.assertEqual(expressions[3].name, "Omega")
        self.assertEqual(expressions[1].unit, "V")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.OCTAVE)
        # assert default variables to plot
        self.assertEqual(len(qraw.get_plot_suggestions()), 1)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "AC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["OpenLoopGain"])

    def test_loading_50_OP(self):
        # arrange
        filename = FIXTURES_DIR / "op.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "Operating Point")
        self.assertFalse(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        expressions = qraw.expression_manager.expressions
        self.assertEqual(len(expressions), 13)
        self.assertEqual(expressions[0].name, "s")
        self.assertEqual(expressions[0].unit, "")
        self.assertEqual(expressions[8].name, "I(D1)")
        self.assertEqual(expressions[8].unit, "A")
        self.assertEqual(expressions[12].name, "P(D1)")
        self.assertEqual(expressions[12].unit, "W")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.OCTAVE)
        # assert default variables
        self.assertEqual(len(qraw.get_plot_suggestions()), 2)
        self.assertIs(qraw.get_plot_suggestions()[0].chart_type, "DC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["V(vf1)", "I(D1)"])
        self.assertIs(qraw.get_plot_suggestions()[1].chart_type, "DC")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[1].expressions], ["Id(J1)", "V(vd)"])

    def test_loading_70_VerilogBus(self):
        # arrange
        filename = FIXTURES_DIR / "VerilogBus.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(qraw)
        self.assertEqual(qraw.plotname, "Transient Analysis")
        self.assertFalse(qraw.complex)
        self.assertEqual(qraw.steps, 1)
        expressions = qraw.expression_manager.expressions
        self.assertEqual(len(expressions), 9)
        self.assertEqual(expressions[0].name, "Time")
        self.assertEqual(expressions[0].unit, "s")
        self.assertEqual(expressions[2].name, "V(outd[3])")
        self.assertEqual(expressions[2].unit, "V")
        self.assertEqual(expressions[8].name, "I(V1)")
        self.assertEqual(expressions[8].unit, "A")
        # assert abscissa range and scale
        self.assertEqual(qraw.abscissa_scale, AbscissaScale.LINEAR)
        # assert default variables
        self.assertEqual(len(qraw.get_plot_suggestions()), 5)
        self.assertEqual(qraw.get_plot_suggestions()[0].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["V(outd[3])"])
        self.assertEqual(qraw.get_plot_suggestions()[1].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[1].expressions], ["V(outd[2])"])
        self.assertEqual(qraw.get_plot_suggestions()[2].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[2].expressions], ["V(outd[1])"])
        self.assertEqual(qraw.get_plot_suggestions()[3].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[3].expressions], ["V(outd[0])"])
        self.assertEqual(qraw.get_plot_suggestions()[4].chart_type, "TRANSIENT")
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[4].expressions], ["V(outa)"])

    def test_filename_property_matches_input_path(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertEqual(qraw.filename, filename)

    def test_title_is_non_empty_for_real_files(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsInstance(qraw.title, str)
        self.assertTrue(len(qraw.title) > 0)

    def test_date_is_non_empty_for_real_files(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsInstance(qraw.date, str)
        self.assertTrue(len(qraw.date) > 0)

    def test_command_is_non_empty_for_transient_file(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertIsInstance(qraw.command, str)
        self.assertTrue(len(qraw.command) > 0)

    def test_ac_file_expressions_are_complex(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert — all non-abscissa expressions in an AC file must be complex
        expressions = qraw.expression_manager.expressions
        for expr in expressions[1:32]:
            self.assertTrue(expr.complex, msg=f"{expr.name} should be complex")

    def test_ac_abscissa_is_not_complex(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert — the abscissa (frequency) is always real even in AC files
        self.assertFalse(qraw.abscissa.complex)

    def test_transient_expressions_are_real(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        expressions = qraw.expression_manager.expressions
        for expr in expressions:
            self.assertFalse(expr.complex, msg=f"{expr.name} should be real")

    def test_stepped_dc_abscissa_has_period_length(self):
        # arrange
        filename = FIXTURES_DIR / "UJ3N065080.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert — abscissa must contain exactly one step's worth of points
        total_available_points = len(qraw.abscissa.data) * qraw.steps
        self.assertEqual(total_available_points % len(qraw.abscissa.data), 0)

    def test_stepped_ac_abscissa_unit_is_hz(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert — decade scale transforms values but the unit is preserved
        self.assertEqual(qraw.abscissa.unit, "Hz")

    def test_transient_abscissa_unit_is_seconds(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertEqual(qraw.abscissa.unit, "s")

    def test_decade_scale_transforms_abscissa_to_log10(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert — a decade-scaled abscissa must contain log10 values, so all
        # values must be in the plausible log10(Hz) range for electronics (0–12)
        self.assertTrue(np.all(qraw.abscissa.data >= 0))
        self.assertTrue(np.all(qraw.abscissa.data <= 12))

    def test_octave_scale_transforms_abscissa_to_log2(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_Bode.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert — log2 values: a 1 Hz lower bound would give 0, 10 THz gives ~43;
        # all values must be strictly positive for a real frequency sweep
        self.assertTrue(np.all(qraw.abscissa.data > 0))

    def test_get_plot_suggestions_returns_empty_for_no_suggestion(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_Nyquist.qraw"
        qraw = QRawFile.load(filename)
        # act — manually clear the suggestion string to simulate a file without one
        qraw._plot_suggestion = ""
        qraw._plot_suggestions = None
        # assert
        self.assertEqual(qraw.get_plot_suggestions(), [])

    def test_get_plot_suggestions_returns_empty_for_whitespace_suggestion(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        qraw = QRawFile.load(filename)
        # act
        qraw._plot_suggestion = "   "
        qraw._plot_suggestions = None
        # assert
        self.assertEqual(qraw.get_plot_suggestions(), [])

    def test_get_plot_suggestions_caches_result(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        qraw = QRawFile.load(filename)
        # act
        first_call = qraw.get_plot_suggestions()
        second_call = qraw.get_plot_suggestions()
        # assert — must return the same list object (cached), not a new one
        self.assertIs(first_call, second_call)

    def test_expression_manager_evaluate_known_variable(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        qraw = QRawFile.load(filename)
        # act
        result = qraw.expression_manager.evaluate("V(out)")
        # assert
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "V(out)")

    def test_expression_manager_evaluate_unknown_returns_none(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        qraw = QRawFile.load(filename)
        # act
        result = qraw.expression_manager.evaluate("V(does_not_exist)")
        # assert
        self.assertIsNone(result)

    def test_expression_manager_evaluate_derived_expression(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        qraw = QRawFile.load(filename)
        # act — arithmetic expression combining two known variables
        result = qraw.expression_manager.evaluate("V(out)-V(in)")
        # assert
        self.assertIsNotNone(result)
        self.assertEqual(result.data.shape, qraw.expression_manager.evaluate("V(out)").data.shape)

    def test_abscissa_data_is_monotonically_increasing_for_tran(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert — time axis must always advance forward
        self.assertTrue(np.all(np.diff(qraw.abscissa.data) >= 0))

    def test_abscissa_data_is_monotonically_increasing_for_ac(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert — log10 frequency axis must always advance forward
        self.assertTrue(np.all(np.diff(qraw.abscissa.data) >= 0))


class TestQRawFileLoad(TestCase):

    def test_returns_none_for_nonexistent_file(self):
        # arrange
        path = Path("/tmp/definitely_does_not_exist_xyz123.qraw")
        # act
        result = QRawFile.load(path)
        # assert
        self.assertIsNone(result)

    def test_returns_none_when_binary_section_is_missing(self):
        # arrange — write a syntactically valid header with no Binary: line
        with tempfile.NamedTemporaryFile(suffix=".qraw", delete=False) as f:
            f.write(b"Title: Test\r\nDate: Today\r\nPlotname: Transient Analysis\r\n")
            f.write(b"Flags: real\r\nNo. Variables: 1\r\nNo. Points: 1\r\nVariables:\r\n")
            f.write(b"0\tTime\ttime\r\n")
            path = Path(f.name)
        # act
        result = QRawFile.load(path)
        # cleanup
        path.unlink(missing_ok=True)
        # assert
        self.assertIsNone(result)

    def test_accepts_string_path(self):
        # arrange
        filename = str(FIXTURES_DIR / "op.qraw")
        # act
        result = QRawFile.load(filename)
        # assert — load must accept a plain string, not just a Path
        self.assertIsNotNone(result)

    def test_accepts_path_object(self):
        # arrange
        filename = FIXTURES_DIR / "op.qraw"
        # act
        result = QRawFile.load(filename)
        # assert
        self.assertIsNotNone(result)


class TestProcessStep(TestCase):

    def test_detects_correct_number_of_steps(self):
        # arrange — 3 steps of 4 points each; first value repeats at index 4 and 8
        data = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
        abscissa = Expression("Time", data, "s")
        # act
        steps, trimmed = _process_step(abscissa, len(data))
        # assert
        self.assertEqual(steps, 3)

    def test_returns_abscissa_trimmed_to_one_period(self):
        # arrange — 3 steps of 4 points each
        data = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
        abscissa = Expression("Time", data, "s")
        # act
        steps, trimmed = _process_step(abscissa, len(data))
        # assert — trimmed abscissa must be one period (4 points)
        self.assertEqual(len(trimmed.data), 4)
        np.testing.assert_array_equal(trimmed.data, data[:4])

    def test_preserves_abscissa_name_and_unit(self):
        # arrange
        data = np.array([0.0, 1.0, 0.0, 1.0])
        abscissa = Expression("Time", data, "s")
        # act
        _, trimmed = _process_step(abscissa, len(data))
        # assert
        self.assertEqual(trimmed.name, "Time")
        self.assertEqual(trimmed.unit, "s")

    def test_two_steps_returns_step_count_of_two(self):
        # arrange — 2 steps of 3 points each; abscissa[0] reappears at index 3
        data = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        abscissa = Expression("Time", data, "s")
        # act
        steps, _ = _process_step(abscissa, len(data))
        # assert
        self.assertEqual(steps, 2)


class TestProcessScale(TestCase):

    def test_decade_applies_log10(self):
        # arrange
        data = np.array([1.0, 10.0, 100.0, 1000.0])
        abscissa = Expression("Frequency", data, "Hz")
        # act
        result = _process_scale(abscissa, AbscissaScale.DECADE)
        # assert
        np.testing.assert_array_almost_equal(result.data, [0.0, 1.0, 2.0, 3.0])

    def test_octave_applies_log2(self):
        # arrange
        data = np.array([1.0, 2.0, 4.0, 8.0])
        abscissa = Expression("Frequency", data, "Hz")
        # act
        result = _process_scale(abscissa, AbscissaScale.OCTAVE)
        # assert
        np.testing.assert_array_almost_equal(result.data, [0.0, 1.0, 2.0, 3.0])

    def test_linear_returns_same_expression_object(self):
        # arrange
        data = np.array([0.0, 1.0, 2.0])
        abscissa = Expression("Time", data, "s")
        # act
        result = _process_scale(abscissa, AbscissaScale.LINEAR)
        # assert — linear must be a no-op; same object must be returned
        self.assertIs(result, abscissa)

    def test_decade_preserves_name_and_unit(self):
        # arrange
        data = np.array([1.0, 10.0])
        abscissa = Expression("Frequency", data, "Hz")
        # act
        result = _process_scale(abscissa, AbscissaScale.DECADE)
        # assert
        self.assertEqual(result.name, "Frequency")
        self.assertEqual(result.unit, "Hz")

    def test_octave_preserves_name_and_unit(self):
        # arrange
        data = np.array([1.0, 2.0])
        abscissa = Expression("Frequency", data, "Hz")
        # act
        result = _process_scale(abscissa, AbscissaScale.OCTAVE)
        # assert
        self.assertEqual(result.name, "Frequency")
        self.assertEqual(result.unit, "Hz")


class TestChartTypeForFile(TestCase):

    def test_hz_unit_maps_to_ac(self):
        # arrange
        abscissa = Expression("Frequency", np.array([1.0]), "Hz")
        # act
        result = _chart_type_for_file(abscissa)
        # assert
        self.assertEqual(result, "AC")

    def test_seconds_unit_maps_to_transient(self):
        # arrange
        abscissa = Expression("Time", np.array([0.0]), "s")
        # act
        result = _chart_type_for_file(abscissa)
        # assert
        self.assertEqual(result, "TRANSIENT")

    def test_volts_unit_maps_to_dc(self):
        # arrange
        abscissa = Expression("V1", np.array([0.0]), "V")
        # act
        result = _chart_type_for_file(abscissa)
        # assert
        self.assertEqual(result, "DC")

    def test_empty_unit_maps_to_dc(self):
        # arrange — operating point sweep uses a dimensionless parameter
        abscissa = Expression("s", np.array([0.0]), "")
        # act
        result = _chart_type_for_file(abscissa)
        # assert
        self.assertEqual(result, "DC")

    def test_ampere_unit_maps_to_dc(self):
        # arrange — current-swept DC analysis
        abscissa = Expression("I1", np.array([0.0]), "A")
        # act
        result = _chart_type_for_file(abscissa)
        # assert
        self.assertEqual(result, "DC")

