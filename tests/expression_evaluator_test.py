from unittest import TestCase

import numpy as np

from viewer.expression import Expression
from viewer.expression_evaluator import ExpressionEvaluator
from viewer.expression_node import FunctionCallNode, NumberNode, VariableRefNode
from viewer.expression_parser import ExpressionParser


class TestExpressionEvaluator(TestCase):

    def test_evaluate_returns_expression_instance(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        self.assertIsInstance(result, Expression)

    def test_evaluate_name_defaults_to_reconstructed_string(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        self.assertEqual(result.name, "V(R1)")

    def test_evaluate_name_override(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1)")
        # act
        result = evaluator.evaluate(tree, context, name="Voltage R1")
        # assert
        self.assertEqual(result.name, "Voltage R1")

    def test_evaluate_source_stored(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1)")
        # act
        result = evaluator.evaluate(tree, context, name="Vr1", source="V(R1)")
        # assert
        self.assertEqual(result.source, "V(R1)")

    def test_evaluate_number_literal(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        tree = parser.parse("5")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertAlmostEqual(float(result.data), 5.0)
        self.assertEqual(result.unit, "")

    def test_evaluate_variable_reference_data(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_equal(result.data, vr1.data)
        self.assertEqual(result.unit, "V")

    def test_evaluate_variable_case_insensitive_lookup(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("v(r1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_equal(result.data, vr1.data)

    def test_evaluate_variable_undefined_raises(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("V(X99)")
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(tree, context)

    def test_evaluate_addition_same_unit(self):
        # arrange — V + V → V
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1) + V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, vr1.data * 2)
        self.assertEqual(result.unit, "V")

    def test_evaluate_subtraction_same_unit(self):
        # arrange — V - V → V
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        vr2 = Expression("V(R2)", np.asarray([0.5, 1.0, 2.0]), "V")
        context = {"V(R1)": vr1, "V(R2)": vr2}
        tree = parser.parse("V(R1) - V(R2)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [0.5, 1.0, 1.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_addition_mixed_units_strips_unit(self):
        # arrange — V + A → dimensionless (no matching unit)
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        ir1 = Expression("I(R1)", np.asarray([0.5, 1.0, 1.5]), "A")
        context = {"V(R1)": vr1, "I(R1)": ir1}
        tree = parser.parse("V(R1) + I(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        self.assertEqual(result.unit, "")

    def test_evaluate_multiplication_voltage_current_gives_power(self):
        # arrange — V * A → W
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        ir1 = Expression("I(R1)", np.asarray([0.5, 1.0, 1.5]), "A")
        context = {"V(R1)": vr1, "I(R1)": ir1}
        tree = parser.parse("V(R1) * I(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        expected = vr1.data * ir1.data
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "W")

    def test_evaluate_scalar_multiplication_preserves_unit(self):
        # arrange — 10 * V → V
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("10 * V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, vr1.data * 10)
        self.assertEqual(result.unit, "V")

    def test_evaluate_division_voltage_over_current_gives_ohm(self):
        # arrange — V / A → Ω
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        ir1 = Expression("I(R1)", np.asarray([0.5, 1.0, 1.5]), "A")
        context = {"V(R1)": vr1, "I(R1)": ir1}
        tree = parser.parse("V(R1) / I(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        expected = vr1.data / ir1.data
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "Ω")

    def test_evaluate_division_current_over_voltage_gives_siemens(self):
        # arrange — A / V → S
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        ir1 = Expression("I(R1)", np.asarray([0.5, 1.0, 1.5]), "A")
        context = {"V(R1)": vr1, "I(R1)": ir1}
        tree = parser.parse("I(R1) / V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        self.assertEqual(result.unit, "S")

    def test_evaluate_division_scalar_over_siemens_gives_ohms(self):
        # arrange — dimensionless / S → Ω (e.g. 1 / conductance = resistance)
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        g = Expression("G", np.asarray([0.5, 1.0, 2.0]), "S")
        context = {"G": g}
        tree = parser.parse("1 / G")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [2.0, 1.0, 0.5])
        self.assertEqual(result.unit, "Ω")

    def test_evaluate_division_scalar_over_ohms_gives_siemens(self):
        # arrange — dimensionless / Ω → S (e.g. 1 / resistance = conductance)
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        r = Expression("R", np.asarray([2.0, 4.0, 8.0]), "Ω")
        context = {"R": r}
        tree = parser.parse("1 / R")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [0.5, 0.25, 0.125])
        self.assertEqual(result.unit, "S")

    def test_evaluate_division_scalar_over_seconds_gives_hz(self):
        # arrange — dimensionless / s → Hz (e.g. 1 / period = frequency)
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        t = Expression("T", np.asarray([0.001, 0.01, 0.1]), "s")
        context = {"T": t}
        tree = parser.parse("1 / T")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [1000.0, 100.0, 10.0])
        self.assertEqual(result.unit, "Hz")

    def test_evaluate_division_scalar_over_hz_gives_seconds(self):
        # arrange — dimensionless / Hz → s (e.g. 1 / frequency = period)
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        f = Expression("F", np.asarray([1.0, 10.0, 100.0]), "Hz")
        context = {"F": f}
        tree = parser.parse("1 / F")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [1.0, 0.1, 0.01])
        self.assertEqual(result.unit, "s")

    def test_evaluate_division_same_unit_gives_dimensionless(self):
        # arrange — V / V → dimensionless
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        vr2 = Expression("V(R2)", np.asarray([2.0, 2.0, 2.0]), "V")
        context = {"V(R1)": vr1, "V(R2)": vr2}
        tree = parser.parse("V(R1) / V(R2)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        self.assertEqual(result.unit, "")

    def test_evaluate_division_by_scalar_preserves_unit(self):
        # arrange — V / scalar → V
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1) / 2")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, vr1.data / 2)
        self.assertEqual(result.unit, "V")

    def test_evaluate_power_operator(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1) ^ 2")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        expected = vr1.data ** 2
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_evaluate_unary_negation(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        tree = parser.parse("-V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, -vr1.data)
        self.assertEqual(result.unit, "V")

    def test_evaluate_abs_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([-1.0, 2.0, -3.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("abs(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_db_function(self):
        # arrange — db(V(out)/V(in)) from the issue examples
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vout = Expression("V(out)", np.asarray([10.0, 100.0]), "V")
        vin = Expression("V(in)", np.asarray([1.0, 1.0]), "V")
        context = {"V(out)": vout, "V(in)": vin}
        tree = parser.parse("db(V(out)/V(in))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [20.0, 40.0])
        self.assertEqual(result.unit, "dB")

    def test_evaluate_sqrt_function(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([4.0, 9.0, 16.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("sqrt(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [2.0, 3.0, 4.0])

    def test_evaluate_complex_variable_db(self):
        # arrange — complex AC variable magnitude in dB
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(out)", np.asarray([1+0j, 10+0j, 100+0j]), "V")
        context = {"V(out)": var}
        tree = parser.parse("db(V(out))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [0.0, 20.0, 40.0])
        self.assertEqual(result.unit, "dB")

    def test_evaluate_real_function_complex_variable(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(out)", np.asarray([3+4j, 1+2j]), "V")
        context = {"V(out)": var}
        tree = parser.parse("real(V(out))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [3.0, 1.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_imag_function_complex_variable(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(out)", np.asarray([3+4j, 1+2j]), "V")
        context = {"V(out)": var}
        tree = parser.parse("imag(V(out))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [4.0, 2.0])

    def test_evaluate_unknown_function_raises(self):
        # arrange
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(R1)": vr1}
        node = FunctionCallNode("xyz", [VariableRefNode("V(R1)")])
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(node, context)

    def test_evaluate_power_expression(self):
        # arrange — instantaneous power: V(R1) * I(R1)
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        ir1 = Expression("I(R1)", np.asarray([0.5, 1.0, 1.5]), "A")
        context = {"V(R1)": vr1, "I(R1)": ir1}
        # act
        tree = parser.parse("V(R1) * I(R1)")
        result = evaluator.evaluate(tree, context, name="Power", source="V(R1) * I(R1)")
        # assert
        expected = vr1.data * ir1.data
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "W")
        self.assertEqual(result.name, "Power")
        self.assertEqual(result.source, "V(R1) * I(R1)")

    def test_evaluate_impedance_expression(self):
        # arrange — impedance: V(R1) / I(R1)
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = Expression("V(R1)", np.asarray([1.0, 2.0, 3.0]), "V")
        ir1 = Expression("I(R1)", np.asarray([0.5, 1.0, 1.5]), "A")
        context = {"V(R1)": vr1, "I(R1)": ir1}
        # act
        tree = parser.parse("V(R1) / I(R1)")
        result = evaluator.evaluate(tree, context)
        # assert
        expected = vr1.data / ir1.data
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "Ω")

    def test_evaluate_alias_omega_expression(self):
        # arrange — from '.alias Omega (2*pi*Frequency)' in VRM_GainBW.qraw / Buck_COT_Bode.qraw
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        freq = Expression("Frequency", np.asarray([1.0, 10.0, 100.0]), "Hz")
        context = {"Frequency": freq}
        tree = parser.parse("(2*pi*Frequency)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [2 * np.pi, 20 * np.pi, 200 * np.pi])
        # 2 (dimensionless) × pi (dimensionless) × Frequency (Hz) → Hz
        self.assertEqual(result.unit, "Hz")

    def test_evaluate_alias_conductance_times_voltage(self):
        # arrange — from '.alias I(R4) (1mho*V(out,0))' in Buck_COT_TRAN.qraw
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        # variable name matches what the parser reconstructs from V(out,0)
        vout = Expression("V(out, 0)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(out, 0)": vout}
        tree = parser.parse("(1mho*V(out,0))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — 1 mho * V equals V numerically (1 S × V gives current in A)
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])
        # 1 (dimensionless) × mho (S) → S; S × V → A
        self.assertEqual(result.unit, "A")

    def test_evaluate_alias_scientific_conductance_times_voltage(self):
        # arrange — from '.alias I(RCOT) (1e-05mho*V(in,n06))' in Buck_COT_TRAN.qraw
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        # variable name matches what the parser reconstructs from V(in,n06)
        vin = Expression("V(in, n06)", np.asarray([100.0, 200.0, 300.0]), "V")
        context = {"V(in, n06)": vin}
        tree = parser.parse("(1e-05mho*V(in,n06))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — 1e-05 S × V gives current scaled by 1e-05
        np.testing.assert_array_almost_equal(result.data, [1e-3, 2e-3, 3e-3])
        # 1e-05 (dimensionless) × mho (S) → S; S × V → A
        self.assertEqual(result.unit, "A")

    def test_evaluate_seconds_constant(self):
        # arrange — built-in time constant from _CONSTANTS: s = 1 second
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        tree = parser.parse("s")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertAlmostEqual(float(result.data), 1.0)
        self.assertEqual(result.unit, "s")

    def test_evaluate_division_by_seconds_constant_gives_hz(self):
        # arrange — 1 / s → Hz using the built-in seconds constant
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        tree = parser.parse("1/s")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertAlmostEqual(float(result.data), 1.0)
        self.assertEqual(result.unit, "Hz")

    def test_evaluate_two_node_probe_differential(self):
        # arrange — V(a, b) = V(a) - V(b) when both single-node probes are in context;
        # models '.alias I(R) (Gmho*V(node_a,node_b))' from real QSPICE output
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        va = Expression("V(node_a)", np.asarray([5.0, 6.0, 7.0]), "V")
        vb = Expression("V(node_b)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(node_a)": va, "V(node_b)": vb}
        tree = parser.parse("V(node_a,node_b)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — differential voltage
        np.testing.assert_array_equal(result.data, [4.0, 4.0, 4.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_two_node_probe_ground_second_arg(self):
        # arrange — V(a, 0) = V(a) - 0 = V(a); node 0 is SPICE ground (not a stored variable);
        # models '.alias I(R323) (0.125mho*V(speaker,0))' from test.qraw
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vspk = Expression("V(speaker)", np.asarray([1.0, -1.0, 2.0]), "V")
        # context only contains V(speaker) — V(0) is absent (SPICE ground is not stored)
        context = {"V(speaker)": vspk}
        tree = parser.parse("V(speaker,0)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — ground subtraction leaves the signal unchanged
        np.testing.assert_array_equal(result.data, [1.0, -1.0, 2.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_two_node_probe_ground_first_arg(self):
        # arrange — V(0, b) = 0 - V(b) = -V(b); node 0 is SPICE ground (not a stored variable);
        # models '.alias I(R_U1_R1•XU305) (G*V(0,u1_n08257•xu305))' from test.qraw
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vb = Expression("V(u1_node)", np.asarray([3.0, 6.0, 9.0]), "V")
        context = {"V(u1_node)": vb}
        tree = parser.parse("V(0,u1_node)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — V(0, b) = -V(b)
        np.testing.assert_array_equal(result.data, [-3.0, -6.0, -9.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_two_node_probe_bullet_node_names(self):
        # arrange — node names containing U+2022 (QSPICE hierarchy separator);
        # models '.alias I(R3•X1•XU) (0.5mho*V(net-a•xu,5•x1•xu))' from test.qraw
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        va = Expression("V(net-a\u2022xu)", np.asarray([10.0, 20.0]), "V")
        vb = Expression("V(5\u2022x1\u2022xu)", np.asarray([2.0, 4.0]), "V")
        context = {"V(net-a\u2022xu)": va, "V(5\u2022x1\u2022xu)": vb}
        tree = parser.parse("V(net-a\u2022xu,5\u2022x1\u2022xu)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — differential voltage across hierarchical nodes
        np.testing.assert_array_equal(result.data, [8.0, 16.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_alias_conductance_times_differential_voltage(self):
        # arrange — full alias pattern from test.qraw: 'G*V(node,0)' where context has V(node)
        # rather than V(node, 0); this is the real-world case the evaluator must handle
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vspk = Expression("V(speaker)", np.asarray([8.0, 16.0, 24.0]), "V")
        context = {"V(speaker)": vspk}
        tree = parser.parse("(0.125mho*V(speaker,0))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — 0.125 S × V(speaker) = current in amperes
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])
        self.assertEqual(result.unit, "A")

    # ------------------------------------------------------------------ #
    # Inverse trigonometric functions                                      #
    # ------------------------------------------------------------------ #

    def test_evaluate_asin_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([0.0, 0.5, 1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("asin(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — unit stripped (returns radians, dimensionless for post-processing)
        np.testing.assert_array_almost_equal(result.data, np.arcsin([0.0, 0.5, 1.0]))
        self.assertEqual(result.unit, "")

    def test_evaluate_arcsin_alias(self):
        # arrange — arcsin is the same function as asin
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([0.0, 1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("arcsin(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, np.arcsin([0.0, 1.0]))

    def test_evaluate_acos_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([1.0, 0.0, -1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("acos(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, np.arccos([1.0, 0.0, -1.0]))
        self.assertEqual(result.unit, "")

    def test_evaluate_arccos_alias(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([0.0, 1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("arccos(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, np.arccos([0.0, 1.0]))

    def test_evaluate_atan_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([0.0, 1.0, -1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("atan(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, np.arctan([0.0, 1.0, -1.0]))
        self.assertEqual(result.unit, "")

    def test_evaluate_arctan_alias(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([0.0, 1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("arctan(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, np.arctan([0.0, 1.0]))

    def test_evaluate_asin_edge_boundary_values(self):
        # arrange — asin(±1) = ±π/2; domain boundary
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([-1.0, 0.0, 1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("asin(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [-np.pi / 2, 0.0, np.pi / 2])

    # ------------------------------------------------------------------ #
    # Two-argument atan2                                                   #
    # ------------------------------------------------------------------ #

    def test_evaluate_atan2_quadrant_values(self):
        # arrange — atan2(y, x) returns radians in (−π, π]
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vy = Expression("V(y)", np.asarray([1.0, -1.0, 0.0, 1.0]), "V")
        vx = Expression("V(x)", np.asarray([1.0, 1.0, -1.0, 0.0]), "V")
        context = {"V(y)": vy, "V(x)": vx}
        tree = parser.parse("atan2(V(y),V(x))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — dimensionless (radians); no unit propagation for atan2
        np.testing.assert_array_almost_equal(result.data, np.arctan2([1.0, -1.0, 0.0, 1.0], [1.0, 1.0, -1.0, 0.0]))
        self.assertEqual(result.unit, "")

    def test_evaluate_atan2_wrong_arg_count_raises(self):
        # arrange — atan2 requires exactly 2 arguments
        evaluator = ExpressionEvaluator()
        node = FunctionCallNode("atan2", [VariableRefNode("V(n)")])
        vn = Expression("V(n)", np.asarray([1.0]), "V")
        context = {"V(n)": vn}
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(node, context)

    # ------------------------------------------------------------------ #
    # Hyperbolic functions                                                 #
    # ------------------------------------------------------------------ #

    def test_evaluate_sinh_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([0.0, 1.0, -1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("sinh(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — unit stripped
        np.testing.assert_array_almost_equal(result.data, np.sinh([0.0, 1.0, -1.0]))
        self.assertEqual(result.unit, "")

    def test_evaluate_cosh_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([0.0, 1.0, -1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("cosh(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — cosh(0) = 1; cosh is symmetric
        np.testing.assert_array_almost_equal(result.data, np.cosh([0.0, 1.0, -1.0]))
        self.assertEqual(result.unit, "")

    def test_evaluate_tanh_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([-2.0, 0.0, 2.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("tanh(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — tanh is bounded in (−1, 1)
        np.testing.assert_array_almost_equal(result.data, np.tanh([-2.0, 0.0, 2.0]))
        self.assertEqual(result.unit, "")

    def test_evaluate_tanh_saturation_edge(self):
        # arrange — tanh(±large) saturates to ±1
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([-1e6, 1e6]), "V")
        context = {"V(n)": var}
        tree = parser.parse("tanh(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [-1.0, 1.0], decimal=5)

    # ------------------------------------------------------------------ #
    # Complex conjugate                                                    #
    # ------------------------------------------------------------------ #

    def test_evaluate_conj_complex_variable(self):
        # arrange — conj preserves both data and unit
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(out)", np.asarray([3+4j, 1-2j]), "V")
        context = {"V(out)": var}
        tree = parser.parse("conj(V(out))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [3-4j, 1+2j])
        self.assertEqual(result.unit, "V")

    def test_evaluate_conj_real_is_identity(self):
        # arrange — conj of a real array is the array itself
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("conj(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])
        self.assertEqual(result.unit, "V")

    # ------------------------------------------------------------------ #
    # ph / phase (aliases for angle)                                       #
    # ------------------------------------------------------------------ #

    def test_evaluate_ph_returns_degrees(self):
        # arrange — purely imaginary → 90°
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(out)", np.asarray([0+1j, 1+0j, -1+0j]), "V")
        context = {"V(out)": var}
        tree = parser.parse("ph(V(out))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [90.0, 0.0, 180.0])
        self.assertEqual(result.unit, "°")

    def test_evaluate_phase_alias_matches_angle(self):
        # arrange — phase() must return identical results to angle()
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(out)", np.asarray([1+1j, -1+1j, -1-1j, 1-1j]), "V")
        context = {"V(out)": var}
        tree_phase = parser.parse("phase(V(out))")
        tree_angle = parser.parse("angle(V(out))")
        # act
        result_phase = evaluator.evaluate(tree_phase, context)
        result_angle = evaluator.evaluate(tree_angle, context)
        # assert
        np.testing.assert_array_almost_equal(result_phase.data, result_angle.data)

    # ------------------------------------------------------------------ #
    # sqr(x) = x^2                                                        #
    # ------------------------------------------------------------------ #

    def test_evaluate_sqr_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([0.0, 2.0, -3.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("sqr(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — unit stripped (would be V², not tracked)
        np.testing.assert_array_almost_equal(result.data, [0.0, 4.0, 9.0])
        self.assertEqual(result.unit, "")

    def test_evaluate_sqr_complex_values(self):
        # arrange — sqr of a complex number is the square, not the magnitude-squared
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(out)", np.asarray([3+4j]), "V")
        context = {"V(out)": var}
        tree = parser.parse("sqr(V(out))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — (3+4j)^2 = 9 + 24j − 16 = −7 + 24j
        np.testing.assert_array_almost_equal(result.data, [(3+4j)**2])

    # ------------------------------------------------------------------ #
    # sign / sgn                                                           #
    # ------------------------------------------------------------------ #

    def test_evaluate_sign_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([-5.0, 0.0, 3.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("sign(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — dimensionless
        np.testing.assert_array_equal(result.data, [-1.0, 0.0, 1.0])
        self.assertEqual(result.unit, "")

    def test_evaluate_sgn_alias(self):
        # arrange — sgn is a QSPICE alias for sign
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([-1.0, 0.0, 1.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("sgn(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_equal(result.data, [-1.0, 0.0, 1.0])

    # ------------------------------------------------------------------ #
    # uramp(x) = max(x, 0)                                                #
    # ------------------------------------------------------------------ #

    def test_evaluate_uramp_clips_negative(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([-3.0, 0.0, 2.0, 5.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("uramp(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — negative values become 0; positive values pass through; unit preserved
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.0, 2.0, 5.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_uramp_all_negative(self):
        # arrange — edge case: all values below zero → all zeros
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([-1.0, -2.0, -100.0]), "V")
        context = {"V(n)": var}
        tree = parser.parse("uramp(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ #
    # Rounding functions                                                   #
    # ------------------------------------------------------------------ #

    def test_evaluate_round_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([1.4, 1.5, 2.5, -1.5]), "V")
        context = {"V(n)": var}
        tree = parser.parse("round(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — numpy uses round-half-to-even; unit preserved
        np.testing.assert_array_almost_equal(result.data, np.round([1.4, 1.5, 2.5, -1.5]))
        self.assertEqual(result.unit, "V")

    def test_evaluate_floor_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([1.9, 2.0, -1.1, -2.9]), "V")
        context = {"V(n)": var}
        tree = parser.parse("floor(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — floor rounds toward −∞; unit preserved
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, -2.0, -3.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_ceil_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([1.1, 2.0, -0.9, -2.1]), "V")
        context = {"V(n)": var}
        tree = parser.parse("ceil(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — ceil rounds toward +∞; unit preserved
        np.testing.assert_array_almost_equal(result.data, [2.0, 2.0, 0.0, -2.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_int_truncates_toward_zero(self):
        # arrange — int(x) truncates toward zero, not toward −∞
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([1.9, -1.9, 2.1, -2.1]), "V")
        context = {"V(n)": var}
        tree = parser.parse("int(V(n))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — int(-1.9) = -1, not -2 (unlike floor); unit preserved
        np.testing.assert_array_almost_equal(result.data, [1.0, -1.0, 2.0, -2.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_int_vs_floor_negative_edge(self):
        # arrange — key distinction: int rounds toward 0, floor rounds toward −∞
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = Expression("V(n)", np.asarray([-0.5, -1.5]), "V")
        context = {"V(n)": var}
        tree_int = parser.parse("int(V(n))")
        tree_floor = parser.parse("floor(V(n))")
        # act
        result_int = evaluator.evaluate(tree_int, context)
        result_floor = evaluator.evaluate(tree_floor, context)
        # assert — int(-0.5) = 0, floor(-0.5) = -1
        np.testing.assert_array_almost_equal(result_int.data, [0.0, -1.0])
        np.testing.assert_array_almost_equal(result_floor.data, [-1.0, -2.0])

    # ------------------------------------------------------------------ #
    # pow / pwr / pwrs                                                     #
    # ------------------------------------------------------------------ #

    def test_evaluate_pow_function(self):
        # arrange — pow(x, y) = x^y, function form of the ^ operator
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vx = Expression("V(x)", np.asarray([2.0, 3.0, 4.0]), "V")
        vy = Expression("V(y)", np.asarray([3.0, 2.0, 0.5]), "")
        context = {"V(x)": vx, "V(y)": vy}
        tree = parser.parse("pow(V(x),V(y))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — unit stripped for power functions
        np.testing.assert_array_almost_equal(result.data, [8.0, 9.0, 2.0])
        self.assertEqual(result.unit, "")

    def test_evaluate_pwr_always_non_negative(self):
        # arrange — pwr(x, y) = |x|^y is always non-negative
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vx = Expression("V(x)", np.asarray([-2.0, -3.0, 2.0]), "V")
        vy = Expression("V(y)", np.asarray([2.0, 3.0, 2.0]), "")
        context = {"V(x)": vx, "V(y)": vy}
        tree = parser.parse("pwr(V(x),V(y))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — pwr(-2, 2) = |-2|^2 = 4; pwr(-3, 3) = |-3|^3 = 27
        np.testing.assert_array_almost_equal(result.data, [4.0, 27.0, 4.0])
        self.assertEqual(result.unit, "")

    def test_evaluate_pwrs_signed_power(self):
        # arrange — pwrs(x, y) = sgn(x) * |x|^y preserves sign of x
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vx = Expression("V(x)", np.asarray([-2.0, 3.0, -4.0]), "V")
        vy = Expression("V(y)", np.asarray([2.0, 2.0, 0.5]), "")
        context = {"V(x)": vx, "V(y)": vy}
        tree = parser.parse("pwrs(V(x),V(y))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — pwrs(-2, 2) = -4; pwrs(3, 2) = 9; pwrs(-4, 0.5) = -2
        np.testing.assert_array_almost_equal(result.data, [-4.0, 9.0, -2.0])
        self.assertEqual(result.unit, "")

    def test_evaluate_pow_wrong_arg_count_raises(self):
        # arrange — pow requires exactly 2 arguments
        evaluator = ExpressionEvaluator()
        node = FunctionCallNode("pow", [VariableRefNode("V(n)")])
        vn = Expression("V(n)", np.asarray([1.0]), "V")
        context = {"V(n)": vn}
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(node, context)

    # ------------------------------------------------------------------ #
    # min / max (two-argument)                                             #
    # ------------------------------------------------------------------ #

    def test_evaluate_min_two_signals(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        va = Expression("V(a)", np.asarray([1.0, 5.0, 3.0]), "V")
        vb = Expression("V(b)", np.asarray([4.0, 2.0, 3.0]), "V")
        context = {"V(a)": va, "V(b)": vb}
        tree = parser.parse("min(V(a),V(b))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — element-wise minimum; first arg's unit preserved
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_max_two_signals(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        va = Expression("V(a)", np.asarray([1.0, 5.0, 3.0]), "V")
        vb = Expression("V(b)", np.asarray([4.0, 2.0, 3.0]), "V")
        context = {"V(a)": va, "V(b)": vb}
        tree = parser.parse("max(V(a),V(b))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — element-wise maximum; first arg's unit preserved
        np.testing.assert_array_almost_equal(result.data, [4.0, 5.0, 3.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_min_all_equal(self):
        # arrange — edge case: all elements equal
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        va = Expression("V(a)", np.asarray([3.0, 3.0, 3.0]), "V")
        vb = Expression("V(b)", np.asarray([3.0, 3.0, 3.0]), "V")
        context = {"V(a)": va, "V(b)": vb}
        tree = parser.parse("min(V(a),V(b))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [3.0, 3.0, 3.0])

    def test_evaluate_min_wrong_arg_count_raises(self):
        # arrange — min requires exactly 2 arguments
        evaluator = ExpressionEvaluator()
        node = FunctionCallNode("min", [VariableRefNode("V(n)")])
        vn = Expression("V(n)", np.asarray([1.0]), "V")
        context = {"V(n)": vn}
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(node, context)

    # ------------------------------------------------------------------ #
    # limit(x, lo, hi)                                                     #
    # ------------------------------------------------------------------ #

    def test_evaluate_limit_clamps_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vx = Expression("V(x)", np.asarray([-5.0, 0.0, 3.0, 10.0]), "V")
        context = {"V(x)": vx}
        tree = parser.parse("limit(V(x),0,5)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — clamped to [0, 5]; first arg's unit preserved
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.0, 3.0, 5.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_limit_all_within_range(self):
        # arrange — edge case: no clamping needed
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vx = Expression("V(x)", np.asarray([1.0, 2.0, 3.0]), "V")
        context = {"V(x)": vx}
        tree = parser.parse("limit(V(x),0,5)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — data passes through unchanged
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_limit_wrong_arg_count_raises(self):
        # arrange — limit requires exactly 3 arguments
        evaluator = ExpressionEvaluator()
        node = FunctionCallNode("limit", [VariableRefNode("V(n)"), NumberNode(0.0)])
        vn = Expression("V(n)", np.asarray([1.0]), "V")
        context = {"V(n)": vn}
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(node, context)
