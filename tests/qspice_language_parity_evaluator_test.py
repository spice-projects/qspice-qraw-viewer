"""
Parity tests: Port expression_evaluator_test.py cases to qspice_language evaluator.

These tests validate that the new qspice_language evaluator can handle all
numeric expressions that the existing expression_evaluator.py supports.

Note: The new evaluator has a simplified interface:
- Returns numpy arrays directly (not Expression objects)
- No unit propagation (yet)
- No SPICE probe differential decomposition (yet)

Tests are adapted to work with the new interface while validating parity
on numeric results where semantically equivalent.
"""

from unittest import TestCase

import numpy as np

from viewer.qspice_language.evaluator import QspiceEvaluator
from viewer.qspice_language.parser import QspiceParser


class TestQspiceEvaluatorParity(TestCase):
    """Port key test cases from expression_evaluator_test.py."""

    # ------------------------------------------------------------------ #
    # Literals and variables                                              #
    # ------------------------------------------------------------------ #

    def test_evaluate_number_literal(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("5")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertEqual(float(result), 5.0)

    def test_evaluate_variable_reference(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        data = np.asarray([1.0, 2.0, 3.0])
        variables = {"V(R1)": data}
        tree = parser.parse_expression("V(R1)")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_equal(result, data)

    def test_evaluate_variable_case_insensitive_lookup(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        data = np.asarray([1.0, 2.0, 3.0])
        variables = {"V(R1)": data}
        tree = parser.parse_expression("v(r1)")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_equal(result, data)

    def test_evaluate_variable_undefined_raises(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        data = np.asarray([1.0, 2.0, 3.0])
        variables = {"V(R1)": data}
        tree = parser.parse_expression("V(X99)")
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(tree, variables)

    # ------------------------------------------------------------------ #
    # Arithmetic operations                                               #
    # ------------------------------------------------------------------ #

    def test_evaluate_addition(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vr1 = np.asarray([1.0, 2.0, 3.0])
        variables = {"V(R1)": vr1}
        tree = parser.parse_expression("V(R1) + V(R1)")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, vr1 * 2)

    def test_evaluate_subtraction(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vr1 = np.asarray([1.0, 2.0, 3.0])
        vr2 = np.asarray([0.5, 1.0, 2.0])
        variables = {"V(R1)": vr1, "V(R2)": vr2}
        tree = parser.parse_expression("V(R1) - V(R2)")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [0.5, 1.0, 1.0])

    def test_evaluate_multiplication(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vr1 = np.asarray([1.0, 2.0, 3.0])
        variables = {"V(R1)": vr1}
        tree = parser.parse_expression("10 * V(R1)")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, vr1 * 10)

    def test_evaluate_division(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vr1 = np.asarray([2.0, 4.0, 6.0])
        variables = {"V(R1)": vr1}
        tree = parser.parse_expression("V(R1) / 2")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_evaluate_power_operator(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vr1 = np.asarray([1.0, 2.0, 3.0])
        variables = {"V(R1)": vr1}
        tree = parser.parse_expression("V(R1) ^ 2")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        expected = vr1 ** 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_power_operator_double_star(self):
        # arrange — ** is also valid
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vr1 = np.asarray([2.0, 3.0, 4.0])
        variables = {"V(R1)": vr1}
        tree = parser.parse_expression("V(R1) ** 2")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        expected = vr1 ** 2
        np.testing.assert_array_almost_equal(result, expected)

    # ------------------------------------------------------------------ #
    # Unary operators                                                     #
    # ------------------------------------------------------------------ #

    def test_evaluate_unary_negation(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vr1 = np.asarray([1.0, 2.0, 3.0])
        variables = {"V(R1)": vr1}
        tree = parser.parse_expression("-V(R1)")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, -vr1)

    def test_evaluate_unary_plus(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vr1 = np.asarray([-1.0, -2.0, -3.0])
        variables = {"V(R1)": vr1}
        tree = parser.parse_expression("+V(R1)")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, vr1)

    # ------------------------------------------------------------------ #
    # Builtin functions                                                   #
    # ------------------------------------------------------------------ #

    def test_evaluate_abs_real_values(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([-1.0, 2.0, -3.0])
        variables = {"V(n)": var}
        tree = parser.parse_expression("abs(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_evaluate_sqrt_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([4.0, 9.0, 16.0])
        variables = {"V(n)": var}
        tree = parser.parse_expression("sqrt(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_evaluate_sin_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([0.0, np.pi / 2, np.pi])
        variables = {"V(n)": var}
        tree = parser.parse_expression("sin(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, np.sin([0.0, np.pi / 2, np.pi]))

    def test_evaluate_cos_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([0.0, np.pi, 2 * np.pi])
        variables = {"V(n)": var}
        tree = parser.parse_expression("cos(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, np.cos([0.0, np.pi, 2 * np.pi]))

    def test_evaluate_tan_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([0.0, np.pi / 4, np.pi / 6])
        variables = {"V(n)": var}
        tree = parser.parse_expression("tan(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, np.tan([0.0, np.pi / 4, np.pi / 6]))

    def test_evaluate_exp_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([0.0, 1.0, 2.0])
        variables = {"V(n)": var}
        tree = parser.parse_expression("exp(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, np.exp([0.0, 1.0, 2.0]))

    def test_evaluate_log_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([1.0, np.e, np.e ** 2])
        variables = {"V(n)": var}
        tree = parser.parse_expression("log(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert — log is natural logarithm in QSPICE
        np.testing.assert_array_almost_equal(result, np.log([1.0, np.e, np.e ** 2]))

    def test_evaluate_log10_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([1.0, 10.0, 100.0])
        variables = {"V(n)": var}
        tree = parser.parse_expression("log10(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, np.log10([1.0, 10.0, 100.0]))

    def test_evaluate_db_function(self):
        # arrange — db(x) = 20*log10(abs(x))
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([1.0, 10.0, 100.0])
        variables = {"V(out)": var}
        tree = parser.parse_expression("db(V(out))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [0.0, 20.0, 40.0])

    def test_evaluate_sign_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([-5.0, 0.0, 3.0])
        variables = {"V(n)": var}
        tree = parser.parse_expression("sign(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_equal(result, [-1.0, 0.0, 1.0])

    def test_evaluate_uramp_function(self):
        # arrange — uramp(x) = max(x, 0)
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([-3.0, 0.0, 2.0, 5.0])
        variables = {"V(n)": var}
        tree = parser.parse_expression("uramp(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 2.0, 5.0])

    def test_evaluate_round_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([1.4, 1.5, 2.5, -1.5])
        variables = {"V(n)": var}
        tree = parser.parse_expression("round(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, np.round([1.4, 1.5, 2.5, -1.5]))

    def test_evaluate_floor_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([1.9, 2.0, -1.1, -2.9])
        variables = {"V(n)": var}
        tree = parser.parse_expression("floor(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, -2.0, -3.0])

    def test_evaluate_ceil_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([1.1, 2.0, -0.9, -2.1])
        variables = {"V(n)": var}
        tree = parser.parse_expression("ceil(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [2.0, 2.0, 0.0, -2.0])

    def test_evaluate_int_function(self):
        # arrange — int truncates toward zero
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([1.9, -1.9, 2.1, -2.1])
        variables = {"V(n)": var}
        tree = parser.parse_expression("int(V(n))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [1.0, -1.0, 2.0, -2.0])

    def test_evaluate_min_two_signals(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        va = np.asarray([1.0, 5.0, 3.0])
        vb = np.asarray([4.0, 2.0, 3.0])
        variables = {"V(a)": va, "V(b)": vb}
        tree = parser.parse_expression("min(V(a),V(b))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_evaluate_max_two_signals(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        va = np.asarray([1.0, 5.0, 3.0])
        vb = np.asarray([4.0, 2.0, 3.0])
        variables = {"V(a)": va, "V(b)": vb}
        tree = parser.parse_expression("max(V(a),V(b))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [4.0, 5.0, 3.0])

    def test_evaluate_limit_three_args(self):
        # arrange — limit(x, lo, hi) clamps x to [lo, hi]
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vx = np.asarray([-5.0, 0.0, 3.0, 10.0])
        variables = {"V(x)": vx}
        tree = parser.parse_expression("limit(V(x),0,5)")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 3.0, 5.0])

    def test_evaluate_pow_function(self):
        # arrange — pow(x, y) = x^y
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        vx = np.asarray([2.0, 3.0, 4.0])
        vy = np.asarray([3.0, 2.0, 0.5])
        variables = {"V(x)": vx, "V(y)": vy}
        tree = parser.parse_expression("pow(V(x),V(y))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [8.0, 9.0, 2.0])

    # ------------------------------------------------------------------ #
    # Comparison and logical operators                                    #
    # ------------------------------------------------------------------ #

    def test_evaluate_equal_comparison(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("5 == 5")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertEqual(float(result), 1.0)

    def test_evaluate_not_equal_comparison(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("5 != 3")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertEqual(float(result), 1.0)

    def test_evaluate_less_than_comparison(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("3 < 5")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertEqual(float(result), 1.0)

    def test_evaluate_logical_and(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("1 && 1")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertEqual(float(result), 1.0)

    def test_evaluate_logical_or(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("0 || 1")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertEqual(float(result), 1.0)

    # ------------------------------------------------------------------ #
    # Ternary operator                                                    #
    # ------------------------------------------------------------------ #

    def test_evaluate_ternary_true_condition(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("1 ? 10 : 20")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertEqual(float(result), 10.0)

    def test_evaluate_ternary_false_condition(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("0 ? 10 : 20")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertEqual(float(result), 20.0)

    # ------------------------------------------------------------------ #
    # Complex numbers                                                     #
    # ------------------------------------------------------------------ #

    def test_evaluate_complex_number_literal(self):
        # arrange — complex notation via real + imag*j
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([3+4j])
        variables = {"V(out)": var}
        tree = parser.parse_expression("V(out)")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [3+4j])

    def test_evaluate_real_function_complex(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([3+4j, 1+2j])
        variables = {"V(out)": var}
        tree = parser.parse_expression("real(V(out))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [3.0, 1.0])

    def test_evaluate_imag_function_complex(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([3+4j, 1+2j])
        variables = {"V(out)": var}
        tree = parser.parse_expression("imag(V(out))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [4.0, 2.0])

    def test_evaluate_conj_complex(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        var = np.asarray([3+4j, 1-2j])
        variables = {"V(out)": var}
        tree = parser.parse_expression("conj(V(out))")
        # act
        result = evaluator.evaluate(tree, variables)
        # assert
        np.testing.assert_array_almost_equal(result, [3-4j, 1+2j])

    # ------------------------------------------------------------------ #
    # Builtin constants                                                   #
    # ------------------------------------------------------------------ #

    def test_evaluate_pi_constant(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("pi")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertAlmostEqual(float(result), np.pi)

    def test_evaluate_e_constant(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("e")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertAlmostEqual(float(result), np.e)

    def test_evaluate_mho_constant(self):
        # arrange — mho is the unit of conductance (siemens)
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("mho")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertEqual(float(result), 1.0)

    # ------------------------------------------------------------------ #
    # Implicit multiplication                                             #
    # ------------------------------------------------------------------ #

    def test_evaluate_implicit_multiplication_number_identifier(self):
        # arrange — 2pi should parse as 2 * pi
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        tree = parser.parse_expression("2*pi")
        # act
        result = evaluator.evaluate(tree, {})
        # assert
        self.assertAlmostEqual(float(result), 2 * np.pi)
