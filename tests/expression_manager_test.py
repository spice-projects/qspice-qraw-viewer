from unittest import TestCase

import numpy as np

from viewer.expression import Expression
from viewer.expression_manager import ExpressionManager


class TestExpressionManager(TestCase):

    def test_expressions_returns_initial_expressions(self):
        # arrange
        expr_a = Expression("V(R1)", np.array([1.0, 2.0, 3.0]), "V")
        expr_b = Expression("I(R1)", np.array([0.1, 0.2, 0.3]), "A")
        manager = ExpressionManager([expr_a, expr_b])
        # act
        result = manager.expressions
        # assert
        self.assertIn(expr_a, result)
        self.assertIn(expr_b, result)
        self.assertEqual(len(result), 2)

    def test_evaluate_returns_existing_expression_by_name(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0, 3.0]), "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("V(R1)")
        # assert — same object must be returned from the cache, not a re-evaluated copy
        self.assertIs(result, expr)

    def test_evaluate_case_insensitive_lookup(self):
        # arrange — expression stored as "V(R1)" should be found via lowercase name
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("v(r1)")
        # assert — same cached object regardless of case used at call site
        self.assertIs(result, expr)

    def test_evaluate_parses_and_evaluates_new_expression(self):
        # arrange
        data = np.array([1.0, 2.0, 3.0])
        expr = Expression("V(R1)", data, "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("2 * V(R1)")
        # assert
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, 2.0 * data)
        self.assertEqual(result.unit, "V")

    def test_evaluate_caches_result_on_second_call(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr])
        # act
        first = manager.evaluate("2 * V(R1)")
        second = manager.evaluate("2 * V(R1)")
        # assert
        self.assertIs(first, second)

    def test_evaluate_returns_none_for_unknown_variable(self):
        # arrange
        manager = ExpressionManager([])
        # act
        result = manager.evaluate("V(unknown)")
        # assert
        self.assertIsNone(result)

    def test_evaluate_returns_none_for_invalid_syntax(self):
        # arrange
        manager = ExpressionManager([])
        # act
        result = manager.evaluate("@@@")
        # assert
        self.assertIsNone(result)

    def test_expressions_includes_newly_evaluated_expression(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr])
        # act
        evaluated = manager.evaluate("2 * V(R1)")
        result = manager.expressions
        # assert
        self.assertIn(evaluated, result)
        self.assertEqual(len(result), 2)

    def test_evaluate_unit_propagation_for_power(self):
        # arrange
        expr_v = Expression("V(R1)", np.array([2.0, 4.0]), "V")
        expr_i = Expression("I(R1)", np.array([1.0, 2.0]), "A")
        manager = ExpressionManager([expr_v, expr_i])
        # act
        result = manager.evaluate("V(R1) * I(R1)")
        # assert
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([2.0, 8.0]))
        self.assertEqual(result.unit, "W")

    def test_evaluate_with_name_stores_result_under_custom_name(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr])
        # act
        evaluated = manager.evaluate("V(R1)", name="output_voltage")
        # assert — result must carry the custom display name
        self.assertEqual(evaluated.name, "output_voltage")

    def test_evaluate_with_name_is_retrievable_by_that_name(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr])
        # act — evaluate once with a name, then retrieve by that name
        first = manager.evaluate("V(R1)", name="vout")
        second = manager.evaluate("vout")
        # assert — same cached object returned on second call
        self.assertIs(first, second)

    def test_evaluate_addition_same_unit_preserves_unit(self):
        # arrange
        expr_a = Expression("V(a)", np.array([1.0, 2.0]), "V")
        expr_b = Expression("V(b)", np.array([0.5, 1.0]), "V")
        manager = ExpressionManager([expr_a, expr_b])
        # act
        result = manager.evaluate("V(a) + V(b)")
        # assert
        self.assertEqual(result.unit, "V")
        np.testing.assert_array_almost_equal(result.data, [1.5, 3.0])

    def test_evaluate_subtraction_same_unit_preserves_unit(self):
        # arrange
        expr_a = Expression("V(a)", np.array([3.0, 5.0]), "V")
        expr_b = Expression("V(b)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr_a, expr_b])
        # act
        result = manager.evaluate("V(a) - V(b)")
        # assert
        self.assertEqual(result.unit, "V")
        np.testing.assert_array_almost_equal(result.data, [2.0, 3.0])

    def test_evaluate_subtraction_different_units_returns_empty_unit(self):
        # arrange
        expr_v = Expression("V(R1)", np.array([1.0]), "V")
        expr_i = Expression("I(R1)", np.array([0.5]), "A")
        manager = ExpressionManager([expr_v, expr_i])
        # act
        result = manager.evaluate("V(R1) - I(R1)")
        # assert — mixed-unit subtraction yields a dimensionless result
        self.assertEqual(result.unit, "")

    def test_evaluate_division_same_units_returns_dimensionless(self):
        # arrange
        expr_a = Expression("V(a)", np.array([4.0, 6.0]), "V")
        expr_b = Expression("V(b)", np.array([2.0, 3.0]), "V")
        manager = ExpressionManager([expr_a, expr_b])
        # act
        result = manager.evaluate("V(a) / V(b)")
        # assert — V/V is dimensionless
        self.assertEqual(result.unit, "")
        np.testing.assert_array_almost_equal(result.data, [2.0, 2.0])

    def test_evaluate_division_V_over_A_produces_ohm(self):
        # arrange
        expr_v = Expression("V(R1)", np.array([10.0, 20.0]), "V")
        expr_i = Expression("I(R1)", np.array([2.0, 4.0]), "A")
        manager = ExpressionManager([expr_v, expr_i])
        # act
        result = manager.evaluate("V(R1) / I(R1)")
        # assert
        self.assertEqual(result.unit, "Ω")
        np.testing.assert_array_almost_equal(result.data, [5.0, 5.0])

    def test_evaluate_division_A_over_V_produces_siemens(self):
        # arrange
        expr_i = Expression("I(R1)", np.array([2.0, 4.0]), "A")
        expr_v = Expression("V(R1)", np.array([10.0, 20.0]), "V")
        manager = ExpressionManager([expr_i, expr_v])
        # act
        result = manager.evaluate("I(R1) / V(R1)")
        # assert
        self.assertEqual(result.unit, "S")

    def test_evaluate_scalar_multiplication_preserves_unit(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("3 * V(R1)")
        # assert — scalar factor is dimensionless; the signal unit must be preserved
        self.assertEqual(result.unit, "V")
        np.testing.assert_array_almost_equal(result.data, [3.0, 6.0])

    def test_evaluate_function_abs_preserves_unit(self):
        # arrange
        expr = Expression("V(R1)", np.array([-1.0, 2.0, -3.0]), "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("abs(V(R1))")
        # assert
        self.assertEqual(result.unit, "V")
        np.testing.assert_array_almost_equal(result.data, [1.0, 2.0, 3.0])

    def test_evaluate_function_db_returns_db_unit(self):
        # arrange — db(V) = 20*log10(|V|)
        expr = Expression("V(R1)", np.array([1.0, 10.0, 100.0]), "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("db(V(R1))")
        # assert
        self.assertEqual(result.unit, "dB")
        np.testing.assert_array_almost_equal(result.data, [0.0, 20.0, 40.0])

    def test_evaluate_function_angle_returns_degrees_unit(self):
        # arrange — angle of a purely real positive number is 0°
        expr = Expression("V(R1)", np.array([1.0 + 0j, 0.0 + 1j]), "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("angle(V(R1))")
        # assert
        self.assertEqual(result.unit, "°")
        np.testing.assert_array_almost_equal(result.data, [0.0, 90.0])

    def test_evaluate_complex_variable_data(self):
        # arrange — AC analysis produces complex-valued expressions
        data = np.array([1.0 + 2j, 3.0 + 4j])
        expr = Expression("V(R1)", data, "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("V(R1)")
        # assert — data and complex flag must be preserved
        self.assertTrue(result.complex)
        np.testing.assert_array_equal(result.data, data)

    def test_evaluate_function_unknown_returns_none(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0]), "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("totally_unknown_func(V(R1))")
        # assert
        self.assertIsNone(result)

    def test_expressions_does_not_grow_on_repeated_evaluate(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr])
        manager.evaluate("2 * V(R1)")
        expected_len = len(manager.expressions)
        # act — calling the same expression a second time must not add a new entry
        manager.evaluate("2 * V(R1)")
        # assert
        self.assertEqual(len(manager.expressions), expected_len)

