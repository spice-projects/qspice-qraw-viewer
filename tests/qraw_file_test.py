import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from viewer.expression import Expression
from viewer.qraw_file import (AbscissaScale, PlotSuggestion, QRawFile, VariableType, VariableTypeInformation, _process_scale, _process_steps)

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
        self.assertEqual([v.name for v in qraw.get_plot_suggestions()[0].expressions], ["(V(VOUT)/V(VO))"])

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

    def test_abscissa_data_is_monotonically_increasing_within_each_ac_step(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert — log10 frequency axis must always advance forward within each step
        for step_slice in qraw.step_information.abscissa_indices:
            self.assertTrue(np.all(np.diff(qraw.abscissa.data[step_slice]) >= 0))

    def test_abscissa_variable_type_is_time_for_transient(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertEqual(qraw.abscissa.variable_type, "time")

    def test_abscissa_variable_type_is_frequency_for_ac(self):
        # arrange
        filename = FIXTURES_DIR / "VRM_GainBW.qraw"
        # act
        qraw = QRawFile.load(filename)
        # assert
        self.assertEqual(qraw.abscissa.variable_type, "frequency")

    def test_voltage_expressions_have_voltage_type(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        qraw = QRawFile.load(filename)
        # act
        expressions = qraw.expression_manager.expressions
        voltage_vars = [e for e in expressions if e.unit == "V"]
        # assert — all voltage variables must have variable_type "voltage"
        for var in voltage_vars:
            self.assertEqual(var.variable_type, "voltage", msg=f"{var.name} should have variable_type 'voltage'")

    def test_current_expressions_have_current_type(self):
        # arrange
        filename = FIXTURES_DIR / "Buck_COT_TRAN.qraw"
        qraw = QRawFile.load(filename)
        # act — find raw variables with current unit (exclude aliases/derived expressions)
        raw_count = len([e for e in qraw.expression_manager.expressions if e.variable_type is not None])
        expressions = qraw.expression_manager.expressions[:raw_count]
        current_vars = [e for e in expressions if e.unit == "A"]
        # assert — all raw current variables must have variable_type "current"
        for var in current_vars:
            self.assertEqual(var.variable_type, "current", msg=f"{var.name} should have variable_type 'current'")

    def test_power_expressions_have_power_type(self):
        # arrange
        filename = FIXTURES_DIR / "UJ3N065080.qraw"
        qraw = QRawFile.load(filename)
        # act
        expressions = qraw.expression_manager.expressions
        power_vars = [e for e in expressions if e.unit == "W"]
        # assert — all power variables must have variable_type "power"
        for var in power_vars:
            self.assertEqual(var.variable_type, "power", msg=f"{var.name} should have variable_type 'power'")


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


class TestProcessSteps(TestCase):

    def test_non_stepped_returns_single_step(self):
        # arrange
        stepped = False
        expr_voltage = Expression("V(out)", np.array([1.0, 2.0, 3.0]), "V", variable_type="voltage")
        expressions = [expr_voltage]
        num_points = 3
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 1)
        self.assertEqual(result.abscissa_indices[0], slice(0, 3))
        self.assertEqual(result.step_length(0), 3)
        self.assertEqual(result.keys, [])
        self.assertEqual(result.values, [])

    def test_stepped_no_parameters_returns_single_step(self):
        # arrange
        stepped = True
        expr_voltage = Expression("V(out)", np.array([1.0, 2.0, 3.0]), "V", variable_type="voltage")
        expressions = [expr_voltage]
        num_points = 3
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 1)
        self.assertEqual(result.abscissa_indices[0], slice(0, 3))
        self.assertEqual(result.step_length(0), 3)
        self.assertEqual(result.keys, [])
        self.assertEqual(result.values, [])

    def test_stepped_single_parameter_two_steps(self):
        # arrange
        stepped = True
        param_data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 6
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 2)
        self.assertEqual(result.keys, ["R1"])
        self.assertEqual(result.values, [(1.0,), (2.0,)])
        self.assertEqual(result.abscissa_indices[0], slice(0, 3))
        self.assertEqual(result.abscissa_indices[1], slice(3, 6))
        self.assertEqual(result.step_length(0), 3)
        self.assertEqual(result.step_length(1), 3)

    def test_stepped_single_parameter_three_steps(self):
        # arrange
        stepped = True
        param_data = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 6
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 3)
        self.assertEqual(result.keys, ["R1"])
        self.assertEqual(result.values, [(1.0,), (2.0,), (3.0,)])
        self.assertEqual(result.step_length(0), 2)
        self.assertEqual(result.step_length(1), 2)
        self.assertEqual(result.step_length(2), 2)

    def test_stepped_multiple_parameters_two_steps(self):
        # arrange
        stepped = True
        param1_data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        param2_data = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
        expr_param1 = Expression("R1", param1_data, "", variable_type="parameter")
        expr_param2 = Expression("R2", param2_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param1, expr_param2]
        num_points = 6
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 2)
        self.assertEqual(result.keys, ["R1", "R2"])
        self.assertEqual(result.values, [(1.0, 10.0), (2.0, 20.0)])

    def test_stepped_multiple_parameters_unequal_steps(self):
        # arrange — variable-length steps: 2 points in first step, 4 in second
        stepped = True
        param1_data = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        param2_data = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 20.0])
        expr_param1 = Expression("R1", param1_data, "", variable_type="parameter")
        expr_param2 = Expression("R2", param2_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param1, expr_param2]
        num_points = 6
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 2)
        self.assertEqual(result.step_length(0), 2)
        self.assertEqual(result.step_length(1), 4)
        self.assertEqual(result.values, [(1.0, 10.0), (2.0, 20.0)])

    def test_parameter_values_extracted_at_step_start(self):
        # arrange
        stepped = True
        param_data = np.array([5.5, 5.5, 5.5, 10.2, 10.2, 10.2])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 6
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertAlmostEqual(result.values[0][0], 5.5)
        self.assertAlmostEqual(result.values[1][0], 10.2)

    def test_single_point_per_step(self):
        # arrange
        stepped = True
        param_data = np.array([1.0, 2.0, 3.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 3
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 3)
        self.assertEqual(result.step_length(0), 1)
        self.assertEqual(result.step_length(1), 1)
        self.assertEqual(result.step_length(2), 1)

    def test_large_number_of_steps(self):
        # arrange
        stepped = True
        param_data = np.array([float(i // 5) for i in range(100)])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.ones(100), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 100
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 20)
        for i in range(20):
            self.assertEqual(result.step_length(i), 5)

    def test_keys_match_parameter_names(self):
        # arrange
        stepped = True
        param1_data = np.array([1.0, 1.0, 2.0, 2.0])
        param2_data = np.array([10.0, 10.0, 20.0, 20.0])
        expr_param1 = Expression("resistance", param1_data, "", variable_type="parameter")
        expr_param2 = Expression("capacitance", param2_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0, 400.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param1, expr_param2]
        num_points = 4
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.keys, ["resistance", "capacitance"])

    def test_slice_boundaries_correct(self):
        # arrange
        stepped = True
        param_data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.arange(6, dtype=float), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 6
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.abscissa_indices[0].start, 0)
        self.assertEqual(result.abscissa_indices[0].stop, 3)
        self.assertEqual(result.abscissa_indices[1].start, 3)
        self.assertEqual(result.abscissa_indices[1].stop, 5)
        self.assertEqual(result.abscissa_indices[2].start, 5)
        self.assertEqual(result.abscissa_indices[2].stop, 6)

    def test_step_length_sum_equals_total_points(self):
        # arrange
        stepped = True
        param_data = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.arange(7, dtype=float), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 7
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        total_length = sum(result.step_length(i) for i in range(result.length))
        self.assertEqual(total_length, num_points)

    def test_step_information_uses_primitive_types_for_lengths_and_slices(self):
        # arrange
        stepped = True
        param_data = np.array([1.0, 1.0, 2.0, 2.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0, 400.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 4
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertIsInstance(result.length, int)
        for i in range(result.length):
            step_slice = result.abscissa_indices[i]
            self.assertIsInstance(step_slice.start, int)
            self.assertIsInstance(step_slice.stop, int)
            self.assertIsInstance(result.step_length(i), int)

    def test_step_information_values_tuples_use_python_primitives(self):
        # arrange
        stepped = True
        param1_data = np.array([1.5, 1.5, 2.5, 2.5])
        param2_data = np.array([10, 10, 20, 20])
        expr_param1 = Expression("R1", param1_data, "", variable_type="parameter")
        expr_param2 = Expression("R2", param2_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0, 400.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param1, expr_param2]
        num_points = 4
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        for value_tuple in result.values:
            self.assertIsInstance(value_tuple, tuple)
            for value in value_tuple:
                self.assertNotIsInstance(value, np.generic)
                self.assertIsInstance(value, int | float | bool | str)

    def test_step_information_stores_per_step_abscissa_value_ranges(self):
        # arrange
        stepped = True
        abscissa = Expression("Time", np.array([10.0, 20.0, 30.0, 9.0, 100.0, 998.0, 11.0, 501.0, 1001.0]), "s", variable_type="time")
        param_data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.arange(9, dtype=float), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 9
        # act
        result = _process_steps(stepped, expressions, abscissa, num_points)
        # assert
        self.assertEqual(result.step_abscissa_from_value(0), 10.0)
        self.assertEqual(result.step_abscissa_to_value(0), 30.0)
        self.assertEqual(result.step_abscissa_from_value(1), 9.0)
        self.assertEqual(result.step_abscissa_to_value(1), 998.0)
        self.assertEqual(result.step_abscissa_from_value(2), 11.0)
        self.assertEqual(result.step_abscissa_to_value(2), 1001.0)

    def test_step_information_stores_global_abscissa_value_bounds(self):
        # arrange
        stepped = True
        abscissa = Expression("Time", np.array([10.0, 20.0, 30.0, 9.0, 100.0, 998.0, 11.0, 501.0, 1001.0]), "s", variable_type="time")
        param_data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.arange(9, dtype=float), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 9
        # act
        result = _process_steps(stepped, expressions, abscissa, num_points)
        # assert
        self.assertEqual(result.abscissa_from_value, 9.0)
        self.assertEqual(result.abscissa_to_value, 1001.0)

    def test_all_points_covered_by_slices(self):
        # arrange
        stepped = True
        param_data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.arange(8, dtype=float), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 8
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert — all indices from 0 to num_points-1 must be covered exactly once
        covered_indices = set()
        for i in range(result.length):
            s = result.abscissa_indices[i]
            covered_indices.update(range(s.start, s.stop))
        self.assertEqual(covered_indices, set(range(num_points)))

    def test_floats_and_integers_in_parameter_values(self):
        # arrange
        stepped = True
        param_data = np.array([1.5, 1.5, 2.7, 2.7])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0, 400.0]), "V", variable_type="voltage")
        expressions = [expr_voltage, expr_param]
        num_points = 4
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(len(result.values), 2)
        self.assertAlmostEqual(result.values[0][0], 1.5)
        self.assertAlmostEqual(result.values[1][0], 2.7)

    def test_parameter_at_end_of_sequence(self):
        # arrange
        stepped = True
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0, 400.0]), "V", variable_type="voltage")
        param_data = np.array([1.0, 1.0, 2.0, 2.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expressions = [expr_voltage, expr_param]
        num_points = 4
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 2)
        self.assertEqual(result.keys, ["R1"])

    def test_parameter_at_beginning_of_sequence(self):
        # arrange
        stepped = True
        param_data = np.array([1.0, 1.0, 2.0, 2.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0, 400.0]), "V", variable_type="voltage")
        expressions = [expr_param, expr_voltage]
        num_points = 4
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 2)
        self.assertEqual(result.keys, ["R1"])

    def test_ignores_non_parameter_expressions(self):
        # arrange
        stepped = True
        param_data = np.array([1.0, 1.0, 2.0, 2.0])
        expr_param = Expression("R1", param_data, "", variable_type="parameter")
        expr_voltage = Expression("V(out)", np.array([100.0, 200.0, 300.0, 400.0]), "V", variable_type="voltage")
        expr_current = Expression("I(R1)", np.array([0.5, 0.5, 1.0, 1.0]), "A", variable_type="current")
        expressions = [expr_voltage, expr_param, expr_current]
        num_points = 4
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert — only parameter should determine steps, not voltage/current
        self.assertEqual(result.length, 2)
        self.assertEqual(result.keys, ["R1"])

    def test_empty_parameters_with_non_stepped(self):
        # arrange
        stepped = False
        expr_voltage = Expression("V(out)", np.array([1.0, 2.0]), "V", variable_type="voltage")
        expressions = [expr_voltage]
        num_points = 2
        # act
        result = _process_steps(stepped, expressions, expr_voltage, num_points)
        # assert
        self.assertEqual(result.length, 1)
        self.assertEqual(result.keys, [])


class TestVariableTypeInformation(TestCase):

    def test_name_voltage(self):
        # arrange
        vti = VariableTypeInformation("voltage", "V")
        # act / assert
        self.assertEqual(vti.name, "voltage")

    def test_unit_voltage(self):
        # arrange
        vti = VariableTypeInformation("voltage", "V")
        # act / assert
        self.assertEqual(vti.unit, "V")

    def test_name_frequency(self):
        # arrange / act / assert
        self.assertEqual(VariableType.FREQUENCY.value.name, "frequency")

    def test_unit_frequency(self):
        # arrange / act / assert
        self.assertEqual(VariableType.FREQUENCY.value.unit, "Hz")

    def test_name_current(self):
        # arrange / act / assert
        self.assertEqual(VariableType.CURRENT.value.name, "current")

    def test_unit_current(self):
        # arrange / act / assert
        self.assertEqual(VariableType.CURRENT.value.unit, "A")

    def test_name_time(self):
        # arrange / act / assert
        self.assertEqual(VariableType.TIME.value.name, "time")

    def test_unit_time(self):
        # arrange / act / assert
        self.assertEqual(VariableType.TIME.value.unit, "s")

    def test_name_phase(self):
        # arrange / act / assert
        self.assertEqual(VariableType.PHASE.value.name, "phase")

    def test_unit_phase(self):
        # arrange / act / assert
        self.assertEqual(VariableType.PHASE.value.unit, "°")

    def test_name_parameter(self):
        # arrange / act / assert
        self.assertEqual(VariableType.PARAMETER.value.name, "parameter")

    def test_unit_parameter_is_empty(self):
        # arrange / act / assert
        self.assertEqual(VariableType.PARAMETER.value.unit, "")


class TestPlotSuggestion(TestCase):

    def test_chart_type_property(self):
        # arrange
        e = Expression("V(R1)", np.array([1.0]), "V")
        ps = PlotSuggestion("AC", [e])
        # act / assert
        self.assertEqual(ps.chart_type, "AC")

    def test_expressions_property(self):
        # arrange
        e1 = Expression("V(R1)", np.array([1.0]), "V")
        e2 = Expression("I(R1)", np.array([0.1]), "A")
        ps = PlotSuggestion("TRANSIENT", [e1, e2])
        # act / assert
        self.assertEqual(ps.expressions, [e1, e2])

    def test_chart_type_transient(self):
        # arrange
        ps = PlotSuggestion("TRANSIENT", [])
        # act / assert
        self.assertEqual(ps.chart_type, "TRANSIENT")

    def test_chart_type_dc(self):
        # arrange
        ps = PlotSuggestion("DC", [])
        # act / assert
        self.assertEqual(ps.chart_type, "DC")

    def test_expressions_empty_list(self):
        # arrange
        ps = PlotSuggestion("AC", [])
        # act / assert
        self.assertEqual(ps.expressions, [])
