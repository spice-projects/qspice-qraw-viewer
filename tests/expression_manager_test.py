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
        # assert
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
