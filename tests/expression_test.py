from unittest import TestCase

import numpy as np

from viewer.expression import Expression


class TestExpression(TestCase):

    def test_name(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        # act
        name = expr.name
        # assert
        self.assertEqual(name, "V(R1)")

    def test_data(self):
        # arrange
        data = np.array([1.0, 2.0, 3.0])
        expr = Expression("V(R1)", data, "V")
        # act
        result = expr.data
        # assert
        np.testing.assert_array_equal(result, data)

    def test_unit(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0]), "V")
        # act
        unit = expr.unit
        # assert
        self.assertEqual(unit, "V")

    def test_source_default_is_none(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0]), "V")
        # act
        source = expr.source
        # assert
        self.assertIsNone(source)

    def test_source_stored(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0]), "V", source="V(R1)")
        # act
        source = expr.source
        # assert
        self.assertEqual(source, "V(R1)")

    def test_complex_false_for_real_data(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        # act
        result = expr.complex
        # assert
        self.assertFalse(result)

    def test_complex_true_for_complex_data(self):
        # arrange
        expr = Expression("V(R1)", np.array([1+2j, 3+4j], dtype=np.complex128), "V")
        # act
        result = expr.complex
        # assert
        self.assertTrue(result)

    def test_values_returns_contiguous_array(self):
        # arrange
        data = np.array([1.0, 2.0, 3.0, 4.0])
        # take a non-contiguous slice (every other element)
        non_contiguous = data[::2]
        expr = Expression("V(R1)", non_contiguous, "V")
        # act
        result = expr.values
        # assert — result must be C-contiguous and contain the same values
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(result, non_contiguous)

    def test_values_caches_result(self):
        # arrange
        expr = Expression("V(R1)", np.array([1.0, 2.0]), "V")
        # act
        first = expr.values
        second = expr.values
        # assert — same object returned on repeated calls
        self.assertIs(first, second)

    def test_values_returns_data_when_already_contiguous(self):
        # arrange
        data = np.ascontiguousarray([1.0, 2.0, 3.0])
        expr = Expression("V(R1)", data, "V")
        # act
        result = expr.values
        # assert — no copy made; same underlying buffer
        self.assertIs(result, data)
