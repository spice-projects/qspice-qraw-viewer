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

    def test_evaluate_function_db_accepts_network_parameter_probe(self):
        # arrange — network parameters such as S11(V1) are stored in the expression context
        expr = Expression("S11(V1)", np.array([1.0 + 0.0j, 0.5 + 0.0j]), "")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("db(S11(V1))")
        # assert
        self.assertIsNotNone(result)
        self.assertEqual(result.unit, "dB")
        np.testing.assert_array_almost_equal(result.data, [0.0, -6.020599913279624])

    def test_evaluate_function_abs_accepts_impedance_probe(self):
        # arrange — impedance parameters should infer ohms even when raw unit metadata is empty
        expr = Expression("Z11(V1)", np.array([50.0 + 0.0j, 75.0 + 0.0j]), "")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("abs(Z11(V1))")
        # assert
        self.assertIsNotNone(result)
        self.assertEqual(result.unit, "Ω")
        np.testing.assert_array_almost_equal(result.data, [50.0, 75.0])

    def test_evaluate_function_abs_accepts_admittance_probe(self):
        # arrange — admittance parameters should infer siemens even when raw unit metadata is empty
        expr = Expression("Y11(V1)", np.array([0.02 + 0.0j, 0.01 + 0.0j]), "")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("abs(Y11(V1))")
        # assert
        self.assertIsNotNone(result)
        self.assertEqual(result.unit, "S")
        np.testing.assert_array_almost_equal(result.data, [0.02, 0.01])

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

    # ------------------------------------------------------------------ #
    # Regression: KiCad-style node names and QSPICE bullet separators   #
    # (test-2.qraw, Transient Analysis, 179 variables)                   #
    # ------------------------------------------------------------------ #

    def test_evaluate_node_name_digit_prefix_bullet_separator_single_probe(self):
        # arrange — QSPICE node names like "7•x1•x1•xu302" start with a digit; the bullet
        # character separates hierarchical levels and must be treated as part of the identifier;
        # using an arithmetic wrapper forces the parser to handle the node name
        expr = Expression("V(7•x1•x1•xu302)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr])
        # act — multiply by 1 to force full expression parsing rather than direct cache lookup
        result = manager.evaluate("1 * V(7•x1•x1•xu302)")
        # assert
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([1.0, 2.0]))

    def test_evaluate_node_name_digit_prefix_bullet_separator_with_ground(self):
        # arrange — alias from log: (1e-09mho*V(7•x1•x1•xu302,0)); second arg is ground
        expr = Expression("V(7•x1•x1•xu302)", np.array([3.0, 6.0]), "V")
        manager = ExpressionManager([expr])
        # act — full conductance alias expression with bullet-separated digit-prefixed node
        result = manager.evaluate("(1e-09mho*V(7•x1•x1•xu302,0))")
        # assert — 1e-9 * V(7•x1•x1•xu302)
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([3e-9, 6e-9]))

    def test_evaluate_node_name_pure_digit_prefix_with_bullet(self):
        # arrange — alias from log: (1e-09mho*V(7•x1•x2•xu302,0)); "7" is the node prefix
        expr = Expression("V(7•x1•x2•xu302)", np.array([2.0, 4.0]), "V")
        manager = ExpressionManager([expr])
        # act
        result = manager.evaluate("V(7•x1•x2•xu302,0)")
        # assert
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([2.0, 4.0]))

    def test_evaluate_node_name_digit_prefix_bullet_in_second_arg(self):
        # arrange — alias from log: (2.5mho*V(16a•x1•xt301,24•xt301)); second arg "24•xt301"
        # starts with the digit 24 followed by bullet, which must be treated as one identifier
        expr_a = Expression("V(16a•x1•xt301)", np.array([5.0, 10.0]), "V")
        expr_b = Expression("V(24•xt301)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr_a, expr_b])
        # act — differential probe where both node names contain bullet separators
        result = manager.evaluate("(2.5mho*V(16a•x1•xt301,24•xt301))")
        # assert — 2.5 * (V(16a•x1•xt301) - V(24•xt301)) = 2.5 * [4, 8]
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([10.0, 20.0]))

    def test_evaluate_kicad_net_name_with_hyphen_underscore_as_probe_arg(self):
        # arrange — KiCad-generated net names like "net-_u304a-g2_" contain hyphens; the parser
        # must treat the whole token as a single identifier, not as arithmetic subtraction
        expr_ot = Expression("V(ot)", np.array([3.0, 6.0]), "V")
        expr_net = Expression("V(net-_u304a-g2_)", np.array([1.0, 2.0]), "V")
        manager = ExpressionManager([expr_ot, expr_net])
        # act — alias from log: (0.01mho*V(ot,net-_u304a-g2_))
        result = manager.evaluate("(0.01mho*V(ot,net-_u304a-g2_))")
        # assert — 0.01 * (V(ot) - V(net-_u304a-g2_)) = 0.01 * [2, 4]
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([0.02, 0.04]))

    def test_evaluate_kicad_net_name_hyphen_underscore_single_probe(self):
        # arrange — KiCad net "net-_u301c-f2_" used as sole V() argument (no differential);
        # using an arithmetic wrapper forces the parser to handle the net name token
        expr = Expression("V(net-_u301c-f2_)", np.array([2.0, 4.0]), "V")
        manager = ExpressionManager([expr])
        # act — multiply by 1 to force full expression parsing rather than direct cache lookup
        result = manager.evaluate("1 * V(net-_u301c-f2_)")
        # assert
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([2.0, 4.0]))

    def test_evaluate_kicad_net_name_both_args_hyphen_pattern(self):
        # arrange — alias from log: (0.00333333333333333mho*V(net-_u302a-g_,net-_u301a-a_))
        expr_a = Expression("V(net-_u302a-g_)", np.array([5.0, 10.0]), "V")
        expr_b = Expression("V(net-_u301a-a_)", np.array([2.0, 3.0]), "V")
        manager = ExpressionManager([expr_a, expr_b])
        # act — both probe arguments are KiCad-style net names
        result = manager.evaluate("(0.00333333333333333mho*V(net-_u302a-g_,net-_u301a-a_))")
        # assert — 1/300 * (V(net-_u302a-g_) - V(net-_u301a-a_)) = 1/300 * [3, 7]
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([0.01, 7.0 / 300.0]))

    def test_evaluate_hierarchical_path_leading_slash_single_probe(self):
        # arrange — QSPICE hierarchical net names start with "/" like "/power_amplifier/vcc1";
        # the slash must be treated as part of the identifier, not as a division operator;
        # using an arithmetic wrapper forces the parser to handle the leading slash
        expr = Expression("V(/power_amplifier/vcc1)", np.array([12.0, 12.0]), "V")
        manager = ExpressionManager([expr])
        # act — multiply by 1 to force full expression parsing rather than direct cache lookup
        result = manager.evaluate("1 * V(/power_amplifier/vcc1)")
        # assert
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([12.0, 12.0]))

    def test_evaluate_hierarchical_path_as_second_probe_argument(self):
        # arrange — alias from log: (5e-05mho*V(vcc,/power_amplifier/vcc1))
        expr_vcc = Expression("V(vcc)", np.array([15.0, 15.0]), "V")
        expr_hier = Expression("V(/power_amplifier/vcc1)", np.array([12.0, 12.0]), "V")
        manager = ExpressionManager([expr_vcc, expr_hier])
        # act — second V() argument is a hierarchical path starting with "/"
        result = manager.evaluate("(5e-05mho*V(vcc,/power_amplifier/vcc1))")
        # assert — 5e-5 * (V(vcc) - V(/power_amplifier/vcc1)) = 5e-5 * 3
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([1.5e-4, 1.5e-4]))

    def test_evaluate_hierarchical_path_as_first_probe_argument(self):
        # arrange — alias from log: (1e-09mho*V(/power_amplifier/katode,0))
        expr = Expression("V(/power_amplifier/katode)", np.array([50.0, 55.0]), "V")
        manager = ExpressionManager([expr])
        # act — first argument is a hierarchical path; second is ground
        result = manager.evaluate("(1e-09mho*V(/power_amplifier/katode,0))")
        # assert — 1e-9 * V(/power_amplifier/katode)
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([5e-8, 5.5e-8]))

    def test_evaluate_hierarchical_path_both_args(self):
        # arrange — alias from log: (0.0001mho*V(/power_amplifier/vcc1,/power_amplifier/vcc2))
        expr_vcc1 = Expression("V(/power_amplifier/vcc1)", np.array([12.0, 12.0]), "V")
        expr_vcc2 = Expression("V(/power_amplifier/vcc2)", np.array([6.0, 6.0]), "V")
        manager = ExpressionManager([expr_vcc1, expr_vcc2])
        # act — both probe arguments are hierarchical paths
        result = manager.evaluate("(0.0001mho*V(/power_amplifier/vcc1,/power_amplifier/vcc2))")
        # assert — 0.0001 * (V(vcc1) - V(vcc2)) = 0.0001 * 6
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([6e-4, 6e-4]))

    def test_evaluate_probe_ground_as_first_argument(self):
        # arrange — alias from log: (1e-09mho*V(0,u1_n08257•xu305)); V(0,node) = -V(node)
        expr = Expression("V(u1_n08257•xu305)", np.array([0.5, 1.0]), "V")
        manager = ExpressionManager([expr])
        # act — first probe argument is ground node "0"; should evaluate as 0 - V(node)
        result = manager.evaluate("(1e-09mho*V(0,u1_n08257•xu305))")
        # assert — 1e-9 * (V(0) - V(u1_n08257•xu305)) = 1e-9 * (-V(node)) = [-5e-10, -1e-9]
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result.data, np.array([-5e-10, -1e-9]))
