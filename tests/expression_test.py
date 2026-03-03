from unittest import TestCase

import numpy as np

from viewer.expression import Expression
from viewer.expression_evaluator import ExpressionEvaluator
from viewer.expression_node import (
    BinaryOp,
    BinaryOpNode,
    FunctionCallNode,
    NumberNode,
    UnaryOp,
    UnaryOpNode,
    VariableRefNode,
)
from viewer.expression_parser import ExpressionParser
from viewer.variable import Variable, VariableType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_var(name: str, vtype: VariableType, values) -> Variable:
    return Variable(0, name, vtype, np.asarray(values, dtype=float))


def _make_complex_var(name: str, vtype: VariableType, values) -> Variable:
    return Variable(0, name, vtype, np.asarray(values, dtype=complex))


# ---------------------------------------------------------------------------
# Expression class tests
# ---------------------------------------------------------------------------

class TestExpression(TestCase):

    def test_name(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        # assert
        self.assertEqual(expr.name, "V(R1)")

    def test_data(self):
        # arrange
        data = np.array([1.0, 2.0, 3.0])
        expr = Expression("V(R1)", data, "V")
        # assert
        np.testing.assert_array_equal(expr.data, data)

    def test_unit(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0]), "V")
        # assert
        self.assertEqual(expr.unit, "V")

    def test_source_default_is_none(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0]), "V")
        # assert
        self.assertIsNone(expr.source)

    def test_source_stored(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0]), "V", source="V(R1)")
        # assert
        self.assertEqual(expr.source, "V(R1)")


# ---------------------------------------------------------------------------
# ExpressionParser tests
# ---------------------------------------------------------------------------

class TestExpressionParser(TestCase):

    def setUp(self):
        self.parser = ExpressionParser()

    # -- Literals --------------------------------------------------------

    def test_parse_integer_literal(self):
        # act
        node = self.parser.parse("10")
        # assert
        self.assertIsInstance(node, NumberNode)
        self.assertEqual(node.value, 10.0)

    def test_parse_float_literal(self):
        # act
        node = self.parser.parse("3.14")
        # assert
        self.assertIsInstance(node, NumberNode)
        self.assertAlmostEqual(node.value, 3.14)

    def test_parse_scientific_notation(self):
        # act
        node = self.parser.parse("1e-3")
        # assert
        self.assertIsInstance(node, NumberNode)
        self.assertAlmostEqual(node.value, 0.001)

    # -- Variable references --------------------------------------------

    def test_parse_bare_identifier(self):
        # act
        node = self.parser.parse("Vout")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "Vout")

    def test_parse_voltage_probe(self):
        # act
        node = self.parser.parse("V(out)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "V(out)")

    def test_parse_voltage_probe_two_nodes(self):
        # act
        node = self.parser.parse("V(R1, 0)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "V(R1, 0)")

    def test_parse_current_probe(self):
        # act
        node = self.parser.parse("I(R1)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "I(R1)")

    def test_parse_device_terminal_current(self):
        # act
        node = self.parser.parse("Id(J1)")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "Id(J1)")

    def test_parse_bus_node_variable(self):
        # act
        node = self.parser.parse("V(outd[3])")
        # assert
        self.assertIsInstance(node, VariableRefNode)
        self.assertEqual(node.name, "V(outd[3])")

    # -- Function calls -------------------------------------------------

    def test_parse_db_function(self):
        # act
        node = self.parser.parse("db(V(out))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name.lower(), "db")
        self.assertEqual(len(node.args), 1)
        self.assertIsInstance(node.args[0], VariableRefNode)
        self.assertEqual(node.args[0].name, "V(out)")

    def test_parse_abs_function(self):
        # act
        node = self.parser.parse("abs(I(R1))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "abs")

    def test_parse_sqrt_function(self):
        # act
        node = self.parser.parse("sqrt(V(in))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name, "sqrt")

    # -- Binary operators -----------------------------------------------

    def test_parse_addition(self):
        # act
        node = self.parser.parse("V(out) + V(in)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.ADD)
        self.assertIsInstance(node.left, VariableRefNode)
        self.assertIsInstance(node.right, VariableRefNode)

    def test_parse_subtraction(self):
        # act
        node = self.parser.parse("V(out) - V(in)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.SUB)

    def test_parse_multiplication(self):
        # act
        node = self.parser.parse("10 * V(R1)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.left, NumberNode)
        self.assertEqual(node.left.value, 10.0)
        self.assertIsInstance(node.right, VariableRefNode)

    def test_parse_division(self):
        # act
        node = self.parser.parse("V(out) / I(R1)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.DIV)

    def test_parse_power(self):
        # act
        node = self.parser.parse("V(out) ^ 2")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.POW)

    # -- Operator precedence --------------------------------------------

    def test_precedence_mul_before_add(self):
        # "a + b * c" should parse as "a + (b * c)"
        # act
        node = self.parser.parse("V(a) + 2 * V(b)")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.ADD)
        self.assertIsInstance(node.right, BinaryOpNode)
        self.assertEqual(node.right.op, BinaryOp.MUL)

    def test_precedence_power_before_mul(self):
        # "a * b ^ 2" should parse as "a * (b ^ 2)"
        # act
        node = self.parser.parse("2 * V(out) ^ 2")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.right, BinaryOpNode)
        self.assertEqual(node.right.op, BinaryOp.POW)

    # -- Unary negation -------------------------------------------------

    def test_parse_unary_negation(self):
        # act
        node = self.parser.parse("-V(out)")
        # assert
        self.assertIsInstance(node, UnaryOpNode)
        self.assertEqual(node.op, UnaryOp.NEG)
        self.assertIsInstance(node.operand, VariableRefNode)

    # -- Parentheses ----------------------------------------------------

    def test_parse_parenthesised_addition(self):
        # act
        node = self.parser.parse("(V(out) + V(in)) * 2")
        # assert
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, BinaryOp.MUL)
        self.assertIsInstance(node.left, BinaryOpNode)
        self.assertEqual(node.left.op, BinaryOp.ADD)

    # -- Complex expressions --------------------------------------------

    def test_parse_db_ratio(self):
        # "db(V(out)/V(in))" — from the issue examples
        # act
        node = self.parser.parse("db(V(out)/V(in))")
        # assert
        self.assertIsInstance(node, FunctionCallNode)
        self.assertEqual(node.name.lower(), "db")
        inner = node.args[0]
        self.assertIsInstance(inner, BinaryOpNode)
        self.assertEqual(inner.op, BinaryOp.DIV)

    # -- Error cases ----------------------------------------------------

    def test_parse_error_unexpected_character(self):
        # assert
        with self.assertRaises(ValueError):
            self.parser.parse("V(out) @ 2")

    def test_parse_error_trailing_token(self):
        # assert
        with self.assertRaises(ValueError):
            self.parser.parse("V(out) 2")

    def test_parse_error_empty_expression(self):
        # assert
        with self.assertRaises(ValueError):
            self.parser.parse("()")


# ---------------------------------------------------------------------------
# ExpressionEvaluator tests
# ---------------------------------------------------------------------------

class TestExpressionEvaluator(TestCase):

    def setUp(self):
        self.parser = ExpressionParser()
        self.evaluator = ExpressionEvaluator()
        # build a small context of test variables
        self.vr1 = _make_var("V(R1)", VariableType.VOLTAGE, [1.0, 2.0, 3.0])
        self.ir1 = _make_var("I(R1)", VariableType.CURRENT, [0.5, 1.0, 1.5])
        self.context: dict[str, Variable] = {
            "V(R1)": self.vr1,
            "I(R1)": self.ir1,
        }

    # -- Expression class output ----------------------------------------

    def test_evaluate_returns_expression_instance(self):
        # arrange
        tree = self.parser.parse("V(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        self.assertIsInstance(result, Expression)

    def test_evaluate_name_defaults_to_reconstructed_string(self):
        # arrange
        tree = self.parser.parse("V(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        self.assertEqual(result.name, "V(R1)")

    def test_evaluate_name_override(self):
        # arrange
        tree = self.parser.parse("V(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context, name="Voltage R1")
        # assert
        self.assertEqual(result.name, "Voltage R1")

    def test_evaluate_source_stored(self):
        # arrange
        tree = self.parser.parse("V(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context, name="Vr1", source="V(R1)")
        # assert
        self.assertEqual(result.source, "V(R1)")

    # -- Numeric literal ------------------------------------------------

    def test_evaluate_number_literal(self):
        # arrange
        tree = self.parser.parse("5")
        # act
        result = self.evaluator.evaluate(tree, {})
        # assert
        self.assertAlmostEqual(float(result.data), 5.0)
        self.assertEqual(result.unit, "")

    # -- Variable reference ---------------------------------------------

    def test_evaluate_variable_reference_data(self):
        # arrange
        tree = self.parser.parse("V(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        np.testing.assert_array_equal(result.data, self.vr1.values)
        self.assertEqual(result.unit, "V")

    def test_evaluate_variable_case_insensitive_lookup(self):
        # arrange
        tree = self.parser.parse("v(r1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        np.testing.assert_array_equal(result.data, self.vr1.values)

    def test_evaluate_variable_undefined_raises(self):
        # arrange
        tree = self.parser.parse("V(X99)")
        # assert
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(tree, self.context)

    # -- Binary operations and unit propagation -------------------------

    def test_evaluate_addition_same_unit(self):
        # arrange — V + V → V
        tree = self.parser.parse("V(R1) + V(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        np.testing.assert_array_almost_equal(result.data, self.vr1.values * 2)
        self.assertEqual(result.unit, "V")

    def test_evaluate_subtraction_same_unit(self):
        # arrange — V - V → V
        vr2 = _make_var("V(R2)", VariableType.VOLTAGE, [0.5, 1.0, 2.0])
        context = {"V(R1)": self.vr1, "V(R2)": vr2}
        tree = self.parser.parse("V(R1) - V(R2)")
        # act
        result = self.evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [0.5, 1.0, 1.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_addition_mixed_units_strips_unit(self):
        # arrange — V + A → dimensionless (no matching unit)
        tree = self.parser.parse("V(R1) + I(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        self.assertEqual(result.unit, "")

    def test_evaluate_multiplication_voltage_current_gives_power(self):
        # arrange — V * A → W
        tree = self.parser.parse("V(R1) * I(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        expected = self.vr1.values * self.ir1.values
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "W")

    def test_evaluate_scalar_multiplication_preserves_unit(self):
        # arrange — 10 * V → V
        tree = self.parser.parse("10 * V(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        np.testing.assert_array_almost_equal(result.data, self.vr1.values * 10)
        self.assertEqual(result.unit, "V")

    def test_evaluate_division_voltage_over_current_gives_ohm(self):
        # arrange — V / A → Ω
        tree = self.parser.parse("V(R1) / I(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        expected = self.vr1.values / self.ir1.values
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "Ω")

    def test_evaluate_division_current_over_voltage_gives_siemens(self):
        # arrange — A / V → S
        tree = self.parser.parse("I(R1) / V(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        self.assertEqual(result.unit, "S")

    def test_evaluate_division_same_unit_gives_dimensionless(self):
        # arrange — V / V → dimensionless
        vr2 = _make_var("V(R2)", VariableType.VOLTAGE, [2.0, 2.0, 2.0])
        context = {"V(R1)": self.vr1, "V(R2)": vr2}
        tree = self.parser.parse("V(R1) / V(R2)")
        # act
        result = self.evaluator.evaluate(tree, context)
        # assert
        self.assertEqual(result.unit, "")

    def test_evaluate_division_by_scalar_preserves_unit(self):
        # arrange — V / scalar → V
        tree = self.parser.parse("V(R1) / 2")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        np.testing.assert_array_almost_equal(result.data, self.vr1.values / 2)
        self.assertEqual(result.unit, "V")

    def test_evaluate_power_operator(self):
        # arrange
        tree = self.parser.parse("V(R1) ^ 2")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        expected = self.vr1.values ** 2
        np.testing.assert_array_almost_equal(result.data, expected)

    # -- Unary negation -------------------------------------------------

    def test_evaluate_unary_negation(self):
        # arrange
        tree = self.parser.parse("-V(R1)")
        # act
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        np.testing.assert_array_almost_equal(result.data, -self.vr1.values)
        self.assertEqual(result.unit, "V")

    # -- Functions -------------------------------------------------------

    def test_evaluate_abs_real_values(self):
        # arrange
        var = _make_var("V(n)", VariableType.VOLTAGE, [-1.0, 2.0, -3.0])
        context = {"V(n)": var}
        tree = self.parser.parse("abs(V(n))")
        # act
        result = self.evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_db_function(self):
        # arrange — db(V(out)/V(in)) from the issue examples
        vout = _make_var("V(out)", VariableType.VOLTAGE, [10.0, 100.0])
        vin  = _make_var("V(in)",  VariableType.VOLTAGE, [1.0,  1.0])
        context = {"V(out)": vout, "V(in)": vin}
        tree = self.parser.parse("db(V(out)/V(in))")
        # act
        result = self.evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [20.0, 40.0])
        self.assertEqual(result.unit, "dB")

    def test_evaluate_sqrt_function(self):
        # arrange
        var = _make_var("V(n)", VariableType.VOLTAGE, [4.0, 9.0, 16.0])
        context = {"V(n)": var}
        tree = self.parser.parse("sqrt(V(n))")
        # act
        result = self.evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [2.0, 3.0, 4.0])

    def test_evaluate_complex_variable_db(self):
        # arrange — complex AC variable magnitude in dB
        var = _make_complex_var("V(out)", VariableType.VOLTAGE, [1+0j, 10+0j, 100+0j])
        context = {"V(out)": var}
        tree = self.parser.parse("db(V(out))")
        # act
        result = self.evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [0.0, 20.0, 40.0])
        self.assertEqual(result.unit, "dB")

    def test_evaluate_real_function_complex_variable(self):
        # arrange
        var = _make_complex_var("V(out)", VariableType.VOLTAGE, [3+4j, 1+2j])
        context = {"V(out)": var}
        tree = self.parser.parse("real(V(out))")
        # act
        result = self.evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [3.0, 1.0])
        self.assertEqual(result.unit, "V")

    def test_evaluate_imag_function_complex_variable(self):
        # arrange
        var = _make_complex_var("V(out)", VariableType.VOLTAGE, [3+4j, 1+2j])
        context = {"V(out)": var}
        tree = self.parser.parse("imag(V(out))")
        # act
        result = self.evaluator.evaluate(tree, context)
        # assert
        np.testing.assert_array_almost_equal(result.data, [4.0, 2.0])

    def test_evaluate_unknown_function_raises(self):
        # arrange — synthesise a FunctionCallNode with an unknown name
        node = FunctionCallNode("xyz", [VariableRefNode("V(R1)")])
        # assert
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(node, self.context)

    # -- End-to-end compound expression ---------------------------------

    def test_evaluate_power_expression(self):
        # "V(R1) * I(R1)" — instantaneous power
        tree = self.parser.parse("V(R1) * I(R1)")
        result = self.evaluator.evaluate(tree, self.context, name="Power", source="V(R1) * I(R1)")
        # assert
        expected = self.vr1.values * self.ir1.values
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "W")
        self.assertEqual(result.name, "Power")
        self.assertEqual(result.source, "V(R1) * I(R1)")

    def test_evaluate_impedance_expression(self):
        # "V(R1) / I(R1)" — impedance
        tree = self.parser.parse("V(R1) / I(R1)")
        result = self.evaluator.evaluate(tree, self.context)
        # assert
        expected = self.vr1.values / self.ir1.values
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(result.unit, "Ω")
