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

    # ------------------------------------------------------------------ #
    # User-defined functions (.func directives)                          #
    # ------------------------------------------------------------------ #

    def test_func_definition_is_callable_in_expressions(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0, 3.0]), "V")
        manager = ExpressionManager([expr], [".func DOUBLE(x) {x * 2}"])
        # act
        result = manager.evaluate("DOUBLE(V(R1))")
        # assert
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, [2.0, 4.0, 6.0])

    def test_func_definition_is_case_insensitive_at_call_site(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr], [".func GAIN(x) {x * 10}"])
        # act
        result = manager.evaluate("gain(V(R1))")
        # assert
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, [10.0, 20.0])

    def test_func_definition_unit_propagation_preserves_voltage(self):
        # arrange — DOUBLE(x) {x * 2}: scaling a voltage must return voltage
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr], [".func DOUBLE(x) {x * 2}"])
        # act
        result = manager.evaluate("DOUBLE(V(R1))")
        # assert
        self.assertEqual(result.unit, "V")

    def test_func_definition_unit_propagation_power(self):
        # arrange — POWER(v, i) {v * i}: V * A should give W
        expr_v = Expression("V(R1)", np.array([2.0, 4.0]), "V")
        expr_i = Expression("I(R1)", np.array([1.0, 2.0]), "A")
        manager = ExpressionManager([expr_v, expr_i], [".func POWER(v, i) {v * i}"])
        # act
        result = manager.evaluate("POWER(V(R1), I(R1))")
        # assert
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, [2.0, 8.0])
        self.assertEqual(result.unit, "W")

    def test_func_definition_calling_another_func(self):
        # arrange — F2 calls F1; both definitions passed together
        expr = Expression("V(R1)", np.array([5.0, 10.0]), "V")
        manager = ExpressionManager([expr], [".func F1(x) {x + 1}", ".func F2(y) {F1(y) * 2}"])
        # act
        result = manager.evaluate("F2(V(R1))")
        # assert — F2(5)=(5+1)*2=12, F2(10)=(10+1)*2=22
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, [12.0, 22.0])

    def test_func_invalid_definition_is_skipped_gracefully(self):
        # arrange — a malformed .func must not prevent the manager from constructing
        expr = Expression("V(R1)", np.array([1.0]), "V")
        manager = ExpressionManager([expr], [".func BAD( {1}"])
        # act — valid expressions must still work
        result = manager.evaluate("V(R1)")
        # assert
        self.assertIsNotNone(result)

    def test_func_unknown_call_after_func_registration_returns_none(self):
        # arrange — only DOUBLE is registered; calling TRIPLE must fail gracefully
        expr = Expression("V(R1)", np.array([1.0]), "V")
        manager = ExpressionManager([expr], [".func DOUBLE(x) {x * 2}"])
        # act
        result = manager.evaluate("TRIPLE(V(R1))")
        # assert
        self.assertIsNone(result)

    # ------------------------------------------------------------------ #
    # Step selectors (@N) via ExpressionManager                          #
    # ------------------------------------------------------------------ #

    def test_step_selector_expression_picks_correct_step(self):
        # arrange — 4-point waveform, two steps of 2 points; @1 should return step 1 rematerialized
        expr = Expression("V(out)", np.array([1.0, 2.0, 10.0, 20.0]), "V")
        step_slices = (slice(0, 2), slice(2, 4))
        manager = ExpressionManager([expr], step_slices=step_slices)
        # act
        result = manager.evaluate("V(out)@1")
        # assert — step 1 is [1, 2]; rematerialized across 2 steps → [1, 2, 1, 2]
        self.assertIsNotNone(result)
        self.assertEqual(len(result.data), 4)
        np.testing.assert_array_almost_equal(result.data, np.array([1.0, 2.0, 1.0, 2.0]))

    def test_step_selector_unit_is_preserved(self):
        # arrange — voltage variable, @N must not change the unit
        expr = Expression("V(out)", np.array([1.0, 2.0, 3.0, 4.0]), "V")
        step_slices = (slice(0, 2), slice(2, 4))
        manager = ExpressionManager([expr], step_slices=step_slices)
        # act
        result = manager.evaluate("V(out)@1")
        # assert
        self.assertEqual(result.unit, "V")

    def test_step_selector_ratio_rematerialized_to_full_length(self):
        # arrange — NFDB-style ratio: step1/step2 per frequency, rematerialized to 4 points
        expr = Expression("V(inoise)", np.array([10.0, 20.0, 2.0, 4.0]), "V")
        step_slices = (slice(0, 2), slice(2, 4))
        manager = ExpressionManager([expr], step_slices=step_slices)
        # act
        result = manager.evaluate("V(inoise)@1 / V(inoise)@2")
        # assert — ratio is [10/2, 20/4] = [5, 5]; tiled to 4 points → [5, 5, 5, 5]
        self.assertIsNotNone(result)
        self.assertEqual(len(result.data), 4)
        np.testing.assert_array_almost_equal(result.data, np.array([5.0, 5.0, 5.0, 5.0]))

    def test_step_selector_func_nfdb_evaluates_to_full_length(self):
        # arrange — real-world NoiseFigure.qraw: .func NFDB(){20*LOG10(V(INOISE)@1/V(INOISE)@2)}
        expr = Expression("V(inoise)", np.array([10.0, 2.0]), "V")
        step_slices = (slice(0, 1), slice(1, 2))
        manager = ExpressionManager([expr], [".func NFDB(){20*LOG10(V(INOISE)@1/V(INOISE)@2)}"], step_slices)
        # act
        result = manager.evaluate("NFDB()")
        # assert — 20*log10(10/2) tiled to 2 points
        self.assertIsNotNone(result)
        self.assertEqual(len(result.data), 2)
        np.testing.assert_array_almost_equal(result.data, np.full(2, 20.0 * np.log10(5.0)))

    def test_step_selector_without_step_slices_returns_none(self):
        # arrange — no step metadata; @N selector cannot be evaluated
        expr = Expression("V(out)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("V(out)@1")
        # assert — must fail gracefully and return None
        self.assertIsNone(result)

    def test_step_selector_non_stepped_expression_unchanged(self):
        # arrange — plain expression without @N must be unaffected by step_slices presence
        expr = Expression("V(R1)", np.array([1.0, 2.0, 3.0, 4.0]), "V")
        step_slices = (slice(0, 2), slice(2, 4))
        manager = ExpressionManager([expr], step_slices=step_slices)
        # act
        result = manager.evaluate("2 * V(R1)")
        # assert — full-length array unchanged
        self.assertIsNotNone(result)
        self.assertEqual(len(result.data), 4)
        np.testing.assert_array_almost_equal(result.data, np.array([2.0, 4.0, 6.0, 8.0]))
