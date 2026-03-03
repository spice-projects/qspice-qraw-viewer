from unittest import TestCase

import numpy as np

from viewer.expression import Expression
from viewer.expression_evaluator import ExpressionEvaluator
from viewer.expression_node import FunctionCallNode, VariableRefNode
from viewer.expression_parser import ExpressionParser
from viewer.variable import Variable, VariableType


def _make_var(name: str, vtype: VariableType, values) -> Variable:
    return Variable(0, name, vtype, np.asarray(values, dtype=float))


def _make_complex_var(name: str, vtype: VariableType, values) -> Variable:
    return Variable(0, name, vtype, np.asarray(values, dtype=complex))


class TestExpressionEvaluator(TestCase):

    def test_evaluate_returns_expression_instance(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
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
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
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
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
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
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
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
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_equal(result.data, vr1.values)
        self.assertEqual(result.unit, "V")

    def test_evaluate_variable_case_insensitive_lookup(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        context = {"V(R1)": vr1}
        tree = parser.parse("v(r1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_equal(result.data, vr1.values)

    def test_evaluate_variable_undefined_raises(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        context = {"V(R1)": vr1}
        tree = parser.parse("V(X99)")
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(tree, context)

    def test_evaluate_addition_same_unit(self):
        # arrange — V + V → V
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1) + V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, vr1.values * 2)
        self.assertEqual(result.unit, "V")

    def test_evaluate_subtraction_same_unit(self):
        # arrange — V - V → V
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        vr2 = _make_var("V(R2)", VariableType.VOLTAGE, [0.5, 1.0, 2.0])
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
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        ir1 = _make_var("I(R1)", VariableType.CURRENT, [0.5, 1.0, 1.5])
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
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        ir1 = _make_var("I(R1)", VariableType.CURRENT, [0.5, 1.0, 1.5])
        context = {"V(R1)": vr1, "I(R1)": ir1}
        tree = parser.parse("V(R1) * I(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        expected = vr1.values * ir1.values
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "W")

    def test_evaluate_scalar_multiplication_preserves_unit(self):
        # arrange — 10 * V → V
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        context = {"V(R1)": vr1}
        tree = parser.parse("10 * V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, vr1.values * 10)
        self.assertEqual(result.unit, "V")

    def test_evaluate_division_voltage_over_current_gives_ohm(self):
        # arrange — V / A → Ω
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        ir1 = _make_var("I(R1)", VariableType.CURRENT, [0.5, 1.0, 1.5])
        context = {"V(R1)": vr1, "I(R1)": ir1}
        tree = parser.parse("V(R1) / I(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        expected = vr1.values / ir1.values
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "Ω")

    def test_evaluate_division_current_over_voltage_gives_siemens(self):
        # arrange — A / V → S
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        ir1 = _make_var("I(R1)", VariableType.CURRENT, [0.5, 1.0, 1.5])
        context = {"V(R1)": vr1, "I(R1)": ir1}
        tree = parser.parse("I(R1) / V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        self.assertEqual(result.unit, "S")

    def test_evaluate_division_same_unit_gives_dimensionless(self):
        # arrange — V / V → dimensionless
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        vr2 = _make_var("V(R2)", VariableType.VOLTAGE, [2.0, 2.0, 2.0])
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
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1) / 2")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, vr1.values / 2)
        self.assertEqual(result.unit, "V")

    def test_evaluate_power_operator(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        context = {"V(R1)": vr1}
        tree = parser.parse("V(R1) ^ 2")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        expected = vr1.values ** 2
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_evaluate_unary_negation(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        context = {"V(R1)": vr1}
        tree = parser.parse("-V(R1)")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, -vr1.values)
        self.assertEqual(result.unit, "V")

    def test_evaluate_abs_real_values(self):
        # arrange
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        var = _make_var("V(n)", VariableType.VOLTAGE, [-1.0, 2.0, -3.0])
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
        vout = _make_var("V(out)", VariableType.VOLTAGE, [10.0, 100.0])
        vin = _make_var("V(in)", VariableType.VOLTAGE, [1.0, 1.0])
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
        var = _make_var("V(n)", VariableType.VOLTAGE, [4.0, 9.0, 16.0])
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
        var = _make_complex_var("V(out)", VariableType.VOLTAGE, [1+0j, 10+0j, 100+0j])
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
        var = _make_complex_var("V(out)", VariableType.VOLTAGE, [3+4j, 1+2j])
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
        var = _make_complex_var("V(out)", VariableType.VOLTAGE, [3+4j, 1+2j])
        context = {"V(out)": var}
        tree = parser.parse("imag(V(out))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [4.0, 2.0])

    def test_evaluate_unknown_function_raises(self):
        # arrange
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        context = {"V(R1)": vr1}
        node = FunctionCallNode("xyz", [VariableRefNode("V(R1)")])
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(node, context)

    def test_evaluate_power_expression(self):
        # arrange — instantaneous power: V(R1) * I(R1)
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        ir1 = _make_var("I(R1)", VariableType.CURRENT, [0.5, 1.0, 1.5])
        context = {"V(R1)": vr1, "I(R1)": ir1}
        # act
        tree = parser.parse("V(R1) * I(R1)")
        result = evaluator.evaluate(tree, context, name="Power", source="V(R1) * I(R1)")
        # assert
        expected = vr1.values * ir1.values
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "W")
        self.assertEqual(result.name, "Power")
        self.assertEqual(result.source, "V(R1) * I(R1)")

    def test_evaluate_impedance_expression(self):
        # arrange — impedance: V(R1) / I(R1)
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        ir1 = _make_var("I(R1)", VariableType.CURRENT, [0.5, 1.0, 1.5])
        context = {"V(R1)": vr1, "I(R1)": ir1}
        # act
        tree = parser.parse("V(R1) / I(R1)")
        result = evaluator.evaluate(tree, context)
        # assert
        expected = vr1.values / ir1.values
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "Ω")

    def test_evaluate_alias_omega_expression(self):
        # arrange — from '.alias Omega (2*pi*Frequency)' in VRM_GainBW.qraw / Buck_COT_Bode.qraw
        parser = ExpressionParser()
        evaluator = ExpressionEvaluator()
        freq = _make_var("Frequency", VariableType.FREQUENCY, [1.0, 10.0, 100.0])
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
        vout = _make_var("V(out, 0)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
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
        vin = _make_var("V(in, n06)", VariableType.VOLTAGE, [100.0, 200.0, 300.0])
        context = {"V(in, n06)": vin}
        tree = parser.parse("(1e-05mho*V(in,n06))")
        # act
        result = evaluator.evaluate(tree, context)
        # assert — 1e-05 S × V gives current scaled by 1e-05
        np.testing.assert_array_almost_equal(result.data, [1e-3, 2e-3, 3e-3])
        # 1e-05 (dimensionless) × mho (S) → S; S × V → A
        self.assertEqual(result.unit, "A")
