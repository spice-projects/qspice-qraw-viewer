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
