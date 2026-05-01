from unittest import TestCase

import numpy as np

from viewer.qspice_language.evaluator import QspiceEvaluator
from viewer.qspice_language.parser import QspiceParser


class TestQspiceEvaluator(TestCase):

    def test_evaluate_number_suffixes(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        expression = parser.parse_expression("1k + 2meg + 3m")
        # act
        result = evaluator.evaluate(expression)
        # assert
        self.assertAlmostEqual(result, 1000.0 + 2000000.0 + 0.003)

    def test_evaluate_identifier_case_insensitive_lookup(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        expression = parser.parse_expression("Gain")
        # act
        result = evaluator.evaluate(expression, variables={"gain": 5.0})
        # assert
        self.assertEqual(result, 5.0)

    def test_evaluate_array_arithmetic(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        expression = parser.parse_expression("vin * 2")
        # act
        result = evaluator.evaluate(expression, variables={"vin": np.asarray([1.0, 2.0, 3.0])})
        # assert
        np.testing.assert_array_equal(result, np.asarray([2.0, 4.0, 6.0]))

    def test_evaluate_relational_expression_returns_numeric_boolean(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        expression = parser.parse_expression("vin > 0")
        # act
        result = evaluator.evaluate(expression, variables={"vin": np.asarray([-1.0, 0.0, 2.0])})
        # assert
        np.testing.assert_array_equal(result, np.asarray([0.0, 0.0, 1.0]))

    def test_evaluate_logical_expression_on_arrays(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        expression = parser.parse_expression("a && b")
        # act
        result = evaluator.evaluate(expression, variables={"a": np.asarray([0.0, 1.0, 1.0]), "b": np.asarray([1.0, 0.0, 2.0])})
        # assert
        np.testing.assert_array_equal(result, np.asarray([0.0, 0.0, 1.0]))

    def test_evaluate_ternary_expression_on_arrays(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        expression = parser.parse_expression("x > 0 ? x : -x")
        # act
        result = evaluator.evaluate(expression, variables={"x": np.asarray([-2.0, 0.0, 3.0])})
        # assert
        np.testing.assert_array_equal(result, np.asarray([2.0, 0.0, 3.0]))

    def test_evaluate_builtin_db_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        expression = parser.parse_expression("db(vout / vin)")
        # act
        result = evaluator.evaluate(expression, variables={"vout": np.asarray([10.0, 100.0]), "vin": np.asarray([1.0, 1.0])})
        # assert
        np.testing.assert_array_almost_equal(result, np.asarray([20.0, 40.0]))

    def test_evaluate_builtin_limit_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        expression = parser.parse_expression("limit(x, -1, 1)")
        # act
        result = evaluator.evaluate(expression, variables={"x": np.asarray([-2.0, 0.5, 3.0])})
        # assert
        np.testing.assert_array_equal(result, np.asarray([-1.0, 0.5, 1.0]))

    def test_evaluate_user_defined_function(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        definition = parser.parse_function_definition(".func gain(x, y) {x / y}")
        expression = parser.parse_expression("gain(vout, vin)")
        # act
        result = evaluator.evaluate(expression, variables={"vout": 10.0, "vin": 2.0}, functions={definition.name: definition})
        # assert
        self.assertEqual(result, 5.0)

    def test_evaluate_recursive_function_raises_error(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        definition = parser.parse_function_definition(".func loop(x) {loop(x)}")
        expression = parser.parse_expression("loop(1)")
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(expression, functions={definition.name: definition})

    def test_evaluate_unknown_identifier_raises_error(self):
        # arrange
        parser = QspiceParser()
        evaluator = QspiceEvaluator()
        expression = parser.parse_expression("missing")
        # act / assert
        with self.assertRaises(ValueError):
            evaluator.evaluate(expression)
